"""
Main entry point for AI Agent Application
Supports multiple interfaces: CLI and HTTP API
"""

import sys
import os
import argparse
from typing import Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Main entry point with interface selection"""
    
    parser = argparse.ArgumentParser(
        description="AI Agent Application - Multiple Interfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py cli --mode simple          # Simple CLI with LangGraph agent
  python main.py cli --mode advanced        # Advanced CLI with agent selection
  python main.py server                     # HTTP API server
  python main.py server --port 8080         # HTTP API server on custom port
  python main.py server --debug             # HTTP API server in debug mode
        """
    )
    
    subparsers = parser.add_subparsers(dest='interface', help='Interface type')
    
    # CLI interface
    cli_parser = subparsers.add_parser('cli', help='Command Line Interface')
    cli_parser.add_argument(
        '--mode', 
        choices=['simple', 'advanced'], 
        default='simple',
        help='CLI mode: simple (single agent) or advanced (agent selection)'
    )
    
    # HTTP server interface
    server_parser = subparsers.add_parser('server', help='HTTP API Server')
    server_parser.add_argument(
        '--host', 
        default='0.0.0.0', 
        help='Host to bind to (default: 0.0.0.0)'
    )
    server_parser.add_argument(
        '--port', 
        type=int, 
        default=8080, 
        help='Port to bind to (default: 8080)'
    )
    server_parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode with auto-reload'
    )
    
    args = parser.parse_args()
    
    # Show help if no interface specified
    if not args.interface:
        parser.print_help()
        print("\n🚀 Welcome to AI Agent Application!")
        print("Choose an interface:")
        print("  • 'cli' - Interactive command line")
        print("  • 'server' - HTTP API server")
        print("\nUse --help with any interface for more options.")
        return
    
    # Handle CLI interface
    if args.interface == 'cli':
        print("🖥️ Starting CLI Interface...")
        
        try:
            from cli_interface import run_simple_cli, run_advanced_cli
            
            if args.mode == 'simple':
                run_simple_cli()
            else:
                run_advanced_cli()
                
        except ImportError as e:
            print(f"❌ Error importing CLI interface: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running CLI: {e}")
            sys.exit(1)
    
    # Handle HTTP server interface
    elif args.interface == 'server':
        print("🌐 Starting HTTP Server Interface...")
        
        try:
            from http_server import run_server
            
            run_server(
                host=args.host,
                port=args.port,
                debug=args.debug
            )
            
        except ImportError as e:
            print(f"❌ Error importing HTTP server: {e}")
            print("Make sure FastAPI and uvicorn are installed:")
            print("pip install fastapi uvicorn[standard]")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running HTTP server: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 