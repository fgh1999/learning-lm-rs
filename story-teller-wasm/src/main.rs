use hyper::{server::conn::Http, service::service_fn, Body, Method, Request, Response, StatusCode};
use lm_infer_core::{
    model::{Llama, ModelResource},
    session::{LmSession, TokenGeneration},
};
use std::{
    net::SocketAddr,
    sync::{Arc, LazyLock},
};
use tokio::net::TcpListener;

static LLAMA: LazyLock<Arc<Llama<f32>>> = LazyLock::new(|| {
    let resources = ModelResource {
        config: Some(include_bytes!(concat!("../../models/", "story", "/config.json")).to_vec()),
        model_data: Some(
            include_bytes!(concat!("../../models/", "story", "/model.safetensors")).to_vec(),
        ),
        ..Default::default()
    };
    Arc::new(Llama::<f32>::from_safetensors(&resources))
});

async fn handle(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    println!("Received request: {:?}", req);
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => Ok(Response::new(Body::from("Hello World!"))),
        (&Method::GET, "/story") => {
            let mut sess = LmSession::new(LLAMA.clone());
            // "Once upon a time, "
            let input_ids = vec![1, 80, 147, 201, 282, 215];
            let output_ids = sess.generate(&input_ids, 500, 0.9, 4, 1., None);

            #[cfg(feature = "perf")]
            sess.print_perf_info();

            Ok(Response::new(Body::from(
                serde_json::to_string(&output_ids).unwrap(),
            )))
        }

        _ => {
            let mut not_found = Response::default();
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // IPv4 address updated from DHCP
    let addr: SocketAddr = "10.0.2.15:80".parse().unwrap();
    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;

        tokio::task::spawn(async move {
            if let Err(err) = Http::new()
                .serve_connection(stream, service_fn(handle))
                .await
            {
                println!("Error serving connection: {:?}", err);
            }
        });
    }
}
