use codex_app_server_protocol::JSONRPCError;
use codex_app_server_protocol::JSONRPCErrorError;
use codex_app_server_protocol::JSONRPCMessage;
use codex_app_server_protocol::JSONRPCRequest;
use codex_app_server_protocol::RemoteControlClientsListParams;
use codex_app_server_protocol::RemoteControlClientsRevokeParams;
use codex_app_server_protocol::RemoteControlDisableParams;
use codex_app_server_protocol::RemoteControlDisableResponse;
use codex_app_server_protocol::RemoteControlEnableParams;
use codex_app_server_protocol::RemoteControlEnableResponse;
use codex_app_server_protocol::RemoteControlPairingStartParams;
use codex_app_server_protocol::RemoteControlPairingStatusParams;
use codex_app_server_protocol::RemoteControlStatusReadResponse;
use codex_app_server_protocol::RequestId;
use codex_app_server_transport::OutgoingMessage;
use codex_app_server_transport::QueuedOutgoingMessage;
use codex_app_server_transport::RemoteControlHandle;
use codex_app_server_transport::RemoteControlPolicy;
use codex_app_server_transport::RemoteControlStartConfig;
use codex_app_server_transport::RemoteControlStartupMode;
use codex_app_server_transport::TransportEvent;
use codex_app_server_transport::start_remote_control;
use codex_core::config::ConfigBuilder;
use codex_core::resolve_installation_id;
use codex_login::AuthManager;
use codex_state::StateRuntime;
use serde::Deserialize;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::io;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

const INVALID_REQUEST: i64 = -32600;
const INVALID_PARAMS: i64 = -32602;
const METHOD_NOT_FOUND: i64 = -32601;
const INTERNAL_ERROR: i64 = -32603;
const RELAY_APP_SERVER_CLIENT_NAME_ENV: &str = "VELLUM_RELAY_APP_SERVER_CLIENT_NAME";

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum RelayInput {
    Message { connection: u64, message: Value },
    Control { message: Value },
}

#[tokio::main]
async fn main() -> io::Result<()> {
    let config = ConfigBuilder::default().build().await?;
    let state_db = StateRuntime::init(
        config.sqlite_config().clone(),
        config.model_provider_id.clone(),
    )
    .await
    .map_err(io::Error::other)?;
    let auth_manager =
        AuthManager::shared_from_config(&config, /*enable_codex_api_key_env*/ false)
            .await
            .map_err(io::Error::other)?;
    let installation_id = resolve_installation_id(&config.codex_home)
        .await
        .map_err(io::Error::other)?;
    // The native stdio App Server keys persisted Remote Control enrollments by
    // the initializing Desktop client's name. The relay runs outside that
    // stdio connection, so its launcher supplies the same identity explicitly
    // and the relay reuses the existing environment instead of creating a
    // second, unpaired server that only differs by an empty client key.
    let app_server_client_name = std::env::var(RELAY_APP_SERVER_CLIENT_NAME_ENV)
        .ok()
        .filter(|name| !name.trim().is_empty());
    let policy = if config
        .config_layer_stack
        .requirements()
        .allow_remote_control
        .as_ref()
        .is_some_and(|requirement| !requirement.value)
    {
        RemoteControlPolicy::DisabledByRequirements
    } else {
        RemoteControlPolicy::Allowed
    };
    let shutdown = CancellationToken::new();
    let (event_tx, mut event_rx) = mpsc::channel(codex_app_server_transport::CHANNEL_CAPACITY);
    let (remote_task, remote_control) = start_remote_control(
        RemoteControlStartConfig {
            remote_control_url: config.chatgpt_base_url.clone(),
            installation_id,
            policy,
        },
        Some(state_db),
        auth_manager,
        event_tx,
        shutdown.clone(),
        /*app_server_client_name_rx*/ None,
        RemoteControlStartupMode::ResolvePersisted,
    )
    .await?;
    let _persisted_enabled = remote_control
        .resolve_persisted_preference(app_server_client_name.as_deref())
        .await
        .unwrap_or(false);

    let (output_tx, mut output_rx) = mpsc::channel::<Value>(128);
    let output_task = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(value) = output_rx.recv().await {
            let mut bytes = serde_json::to_vec(&value).map_err(io::Error::other)?;
            bytes.push(b'\n');
            stdout.write_all(&bytes).await?;
            stdout.flush().await?;
        }
        Ok::<(), io::Error>(())
    });

    let mut status_rx = remote_control.status_receiver();
    let status_tx = output_tx.clone();
    status_rx.borrow_and_update();
    let status_task = tokio::spawn(async move {
        while status_rx.changed().await.is_ok() {
            let status = status_rx.borrow_and_update().clone();
            if status_tx
                .send(json!({"kind":"status","status":status}))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    send_output(&output_tx, json!({"kind":"ready"})).await?;

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut writers: HashMap<u64, mpsc::Sender<QueuedOutgoingMessage>> = HashMap::new();
    loop {
        tokio::select! {
            line = lines.next_line() => {
                let Some(line) = line? else { break };
                let input = match serde_json::from_str::<RelayInput>(&line) {
                    Ok(input) => input,
                    Err(_) => continue,
                };
                match input {
                    RelayInput::Message { connection, message } => {
                        if let Some(writer) = writers.get(&connection) {
                            let queued = QueuedOutgoingMessage::new(
                                OutgoingMessage::RawNotification(message),
                            );
                            let _sent = writer.send(queued).await;
                        }
                    }
                    RelayInput::Control { message } => {
                        let response = handle_control(
                            &remote_control,
                            message,
                            app_server_client_name.as_deref(),
                        )
                        .await;
                        send_output(&output_tx, json!({"kind":"control","message":response})).await?;
                    }
                }
            }
            event = event_rx.recv() => {
                let Some(event) = event else { break };
                match event {
                    TransportEvent::ConnectionOpened { connection_id, writer, .. } => {
                        writers.insert(connection_id.0, writer);
                        send_output(&output_tx, json!({"kind":"open","connection":connection_id.0})).await?;
                    }
                    TransportEvent::ConnectionClosed { connection_id } => {
                        writers.remove(&connection_id.0);
                        send_output(&output_tx, json!({"kind":"close","connection":connection_id.0})).await?;
                    }
                    TransportEvent::IncomingMessage { connection_id, message } => {
                        send_output(&output_tx, json!({
                            "kind":"message",
                            "connection":connection_id.0,
                            "message":message,
                        })).await?;
                    }
                }
            }
        }
    }

    shutdown.cancel();
    let _remote_result = remote_task.await;
    status_task.abort();
    drop(output_tx);
    output_task.await.map_err(io::Error::other)??;
    Ok(())
}

async fn send_output(output_tx: &mpsc::Sender<Value>, value: Value) -> io::Result<()> {
    output_tx
        .send(value)
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "relay output closed"))
}

async fn handle_control(
    handle: &RemoteControlHandle,
    message: Value,
    app_server_client_name: Option<&str>,
) -> Value {
    let request = match serde_json::from_value::<JSONRPCMessage>(message) {
        Ok(JSONRPCMessage::Request(request)) => request,
        Ok(
            JSONRPCMessage::Notification(_)
            | JSONRPCMessage::Response(_)
            | JSONRPCMessage::Error(_),
        )
        | Err(_) => {
            return serde_json::to_value(JSONRPCError {
                id: RequestId::Integer(0),
                error: rpc_error(INVALID_REQUEST, "relay control expects a JSON-RPC request"),
            })
            .unwrap_or_else(|_| json!({"id":0,"error":{"code":INTERNAL_ERROR,"message":"serialization failed"}}));
        }
    };
    let id = request.id.clone();
    match handle_control_request(handle, request, app_server_client_name).await {
        Ok(result) => json!({"id":id,"result":result}),
        Err(error) => serde_json::to_value(JSONRPCError { id, error }).unwrap_or_else(
            |_| json!({"id":0,"error":{"code":INTERNAL_ERROR,"message":"serialization failed"}}),
        ),
    }
}

async fn handle_control_request(
    handle: &RemoteControlHandle,
    request: JSONRPCRequest,
    app_server_client_name: Option<&str>,
) -> Result<Value, JSONRPCErrorError> {
    let params = request.params.unwrap_or(Value::Null);
    match request.method.as_str() {
        "remoteControl/enable" => {
            let params: Option<RemoteControlEnableParams> = parse_params(params)?;
            let params = params.unwrap_or_default();
            let status = if params.ephemeral {
                handle
                    .enable_ephemeral()
                    .map_err(|error| map_io(error.to_string()))?
            } else {
                handle
                    .enable(app_server_client_name)
                    .await
                    .map_err(map_io)?
            };
            serde_json::to_value(RemoteControlEnableResponse::from(status)).map_err(map_serde)
        }
        "remoteControl/disable" => {
            let params: Option<RemoteControlDisableParams> = parse_params(params)?;
            let params = params.unwrap_or_default();
            let status = if params.ephemeral {
                handle.disable_ephemeral().await
            } else {
                handle
                    .disable(app_server_client_name)
                    .await
                    .map_err(map_io)?
            };
            serde_json::to_value(RemoteControlDisableResponse::from(status)).map_err(map_serde)
        }
        "remoteControl/status/read" => {
            let status = handle.status();
            serde_json::to_value(RemoteControlStatusReadResponse {
                status: status.status,
                server_name: status.server_name,
                installation_id: status.installation_id,
                environment_id: status.environment_id,
            })
            .map_err(map_serde)
        }
        "remoteControl/pairing/start" => {
            let params = parse_params::<RemoteControlPairingStartParams>(params)?;
            let response = handle
                .start_pairing(params, app_server_client_name)
                .await
                .map_err(map_io)?;
            serde_json::to_value(response).map_err(map_serde)
        }
        "remoteControl/pairing/status" => {
            let params = parse_params::<RemoteControlPairingStatusParams>(params)?;
            match (&params.pairing_code, &params.manual_pairing_code) {
                (Some(_), None) | (None, Some(_)) => {}
                _ => {
                    return Err(rpc_error(
                        INVALID_PARAMS,
                        "pairing status requires exactly one pairing code",
                    ));
                }
            }
            let response = handle.pairing_status(params).await.map_err(map_io)?;
            serde_json::to_value(response).map_err(map_serde)
        }
        "remoteControl/client/list" => {
            let response = handle
                .list_clients(parse_params::<RemoteControlClientsListParams>(params)?)
                .await
                .map_err(map_io)?;
            serde_json::to_value(response).map_err(map_serde)
        }
        "remoteControl/client/revoke" => {
            let response = handle
                .revoke_client(parse_params::<RemoteControlClientsRevokeParams>(params)?)
                .await
                .map_err(map_io)?;
            serde_json::to_value(response).map_err(map_serde)
        }
        method => Err(rpc_error(
            METHOD_NOT_FOUND,
            format!("unsupported relay control method: {method}"),
        )),
    }
}

fn parse_params<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, JSONRPCErrorError> {
    serde_json::from_value(value).map_err(|error| rpc_error(INVALID_PARAMS, error.to_string()))
}

fn map_io(error: impl std::fmt::Display) -> JSONRPCErrorError {
    rpc_error(INTERNAL_ERROR, error.to_string())
}

fn map_serde(error: serde_json::Error) -> JSONRPCErrorError {
    rpc_error(INTERNAL_ERROR, error.to_string())
}

fn rpc_error(code: i64, message: impl Into<String>) -> JSONRPCErrorError {
    JSONRPCErrorError {
        code,
        message: message.into(),
        data: None,
    }
}
