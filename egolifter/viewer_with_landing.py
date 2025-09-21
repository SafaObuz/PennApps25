#!/usr/bin/env python3
"""
Wrapper script to run the EgoLifter viewer with a landing page.
This script starts both the Viser viewer and serves a landing page that embeds it.
"""

import http.server
import socketserver
import threading
import time
import webbrowser
import argparse
import os
import sys
import subprocess
from pathlib import Path

class LandingPageHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve the landing page"""
    
    def do_GET(self):
        if self.path == '/' or self.path == '':
            # Serve the landing page
            self.path = '/landing_page.html'
        return super().do_GET()
    
    def end_headers(self):
        # Add headers to allow iframe embedding
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def start_landing_server(port=8081):
    """Start the landing page server"""
    # Change to the directory containing landing_page.html
    os.chdir(Path(__file__).parent)
    
    handler = LandingPageHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Landing page server running at http://localhost:{port}")
        httpd.serve_forever()

def start_viewer(model_root, viewer_port=8080, args=None):
    """Start the Viser viewer"""
    cmd = [
        sys.executable,
        "viewer.py",
        model_root,
        "--port", str(viewer_port),
        "--host", "0.0.0.0"
    ]
    
    # Add additional arguments if provided
    if args:
        if args.image_format:
            cmd.extend(["--image_format", args.image_format])
        if args.reorient:
            cmd.extend(["--reorient", args.reorient])
        if args.enable_transform:
            cmd.append("--enable_transform")
        if args.show_cameras:
            cmd.append("--show_cameras")
        if args.data_root:
            cmd.extend(["--data_root", args.data_root])
        if args.feat_pca:
            cmd.append("--feat_pca")
    
    print(f"Starting viewer with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
    return process

def main():
    parser = argparse.ArgumentParser(description="Run EgoLifter viewer with landing page")
    parser.add_argument("model_root", type=str, help="Path to the model root directory")
    parser.add_argument("--landing_port", type=int, default=8081, 
                      help="Port for the landing page server (default: 8081)")
    parser.add_argument("--viewer_port", type=int, default=8080,
                      help="Port for the Viser viewer (default: 8080)")
    parser.add_argument("--no_browser", action="store_true",
                      help="Don't automatically open browser")
    
    # Pass through viewer arguments
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                      help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--enable_transform", "--enable-transform",
                      action="store_true", default=False,
                      help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras", action="store_true")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--feat_pca", action="store_true", help="Enable PCA feature visualization")
    
    args = parser.parse_args()
    
    # First, update the landing page HTML to use the correct viewer port
    landing_html_path = Path(__file__).parent / "landing_page.html"
    if landing_html_path.exists():
        with open(landing_html_path, 'r') as f:
            content = f.read()
        # Update the iframe src to use the correct port
        content = content.replace('http://localhost:8080', f'http://localhost:{args.viewer_port}')
        with open(landing_html_path, 'w') as f:
            f.write(content)
    
    # Start the viewer process
    print(f"Starting EgoLifter viewer on port {args.viewer_port}...")
    viewer_process = start_viewer(args.model_root, args.viewer_port, args)
    
    # Give the viewer time to start
    time.sleep(3)
    
    # Start the landing page server in a separate thread
    print(f"Starting landing page server on port {args.landing_port}...")
    landing_thread = threading.Thread(
        target=start_landing_server, 
        args=(args.landing_port,),
        daemon=True
    )
    landing_thread.start()
    
    # Give the server time to start
    time.sleep(1)
    
    # Open browser if requested
    if not args.no_browser:
        url = f"http://localhost:{args.landing_port}"
        print(f"Opening browser at {url}")
        webbrowser.open(url)
    
    print("\n" + "="*60)
    print("EgoLifter Scene Editor is running!")
    print(f"Landing page: http://localhost:{args.landing_port}")
    print(f"Direct viewer: http://localhost:{args.viewer_port}")
    print("Press Ctrl+C to stop both servers")
    print("="*60 + "\n")
    
    try:
        # Keep the main thread alive
        viewer_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        viewer_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()
