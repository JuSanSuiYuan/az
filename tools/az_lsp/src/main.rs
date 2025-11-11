use tower_lsp::{LspService, Server};

mod server;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Create LSP service
    let (service, socket) = LspService::new(|client| server::AzLspServer::new(client));

    // Start server
    tracing::info!("Starting AZ lsp server");
    Server::new(tokio::io::stdin(), tokio::io::stdout(), socket)
        .serve(service)
        .await;
}
