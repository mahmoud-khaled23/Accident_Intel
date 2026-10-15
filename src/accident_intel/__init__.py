"""Command-line entry point for the API server."""
import argparse


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the Accident Intel API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)
