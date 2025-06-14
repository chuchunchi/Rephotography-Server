#!/usr/bin/env python3
import socket
import time
import sys
import subprocess

def check_port(host, port):
    # Try to run netstat to see if the port is in use
    try:
        result = subprocess.run(['netstat', '-an'], capture_output=True, text=True)
        if str(port) in result.stdout:
            print(f"[Info] Port {port} appears to be in use locally")
    except:
        pass

def test_connection(host='192.168.0.158', ports=[8080, 8081, 8082, 9999]):
    # First check if we can ping the host
    print(f"\n[Test] Checking if host {host} is reachable...")
    try:
        ping_result = subprocess.run(['ping', '-c', '1', '-W', '2', host], 
                                   capture_output=True, text=True)
        if ping_result.returncode == 0:
            print(f"[Test] Host {host} is reachable")
        else:
            print(f"[Error] Cannot ping host {host}")
            return
    except Exception as e:
        print(f"[Error] Ping test failed: {e}")

    # Try each port
    for port in ports:
        print(f"\n[Test] Attempting to connect to {host}:{port}")
        check_port(host, port)
        
        try:
            # Create socket with debugging
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)  # 3 second timeout
            
            # Try to set socket options for debugging
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Connect
            print(f"[Test] Connecting to port {port}...")
            result = sock.connect_ex((host, port))
            
            if result != 0:
                print(f"[Error] Connection failed with error code {result}")
                if result == 111:
                    print("[Info] Error 111 means no service is listening on this port")
                elif result == 113:
                    print("[Info] Error 113 means no route to host")
                elif result == 110:
                    print("[Info] Error 110 means connection timed out")
                continue
                
            print(f"[Test] Connected successfully to port {port}!")
            
            # Test commands
            test_commands = [
                "1,TakeoffOrLanding,0,0,0,0",      # Takeoff/Landing command
                "2,Custom,0.2,0,0,0",              # Move right with speed 0.2
                "1,yawRight,0,0,5,0",              # Turn right 5 degrees
                "1,up,0,0,0,0.2",                  # Move up with speed 0.2
            ]
            
            for cmd in test_commands:
                try:
                    print(f"\n[Test] Sending command: {cmd}")
                    sock.send(cmd.encode())
                    time.sleep(1)
                    
                    try:
                        response = sock.recv(1024)
                        if response:
                            print(f"[Test] Received response: {response.decode()}")
                    except socket.timeout:
                        print("[Info] No response received (timeout)")
                    
                except Exception as e:
                    print(f"[Error] Failed to send command: {e}")
            
            print("\n[Test] All commands sent successfully!")
            
        except Exception as e:
            print(f"[Error] Connection failed: {e}")
        
        finally:
            try:
                sock.close()
                print(f"[Test] Connection to port {port} closed")
            except:
                pass

if __name__ == "__main__":
    # Allow command line arguments for host and port
    HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.0.158"
    PORTS = [int(p) for p in sys.argv[2].split(',')] if len(sys.argv) > 2 else [8080, 8081, 8082, 9999]
    
    print("=== Drone Control TCP Connection Test ===")
    print("This script will:")
    print("1. Check if host is reachable (ping)")
    print("2. Try to connect to multiple ports")
    print("3. Send test commands if connected")
    print(f"\nTesting host: {HOST}")
    print(f"Testing ports: {PORTS}")
    print("\nCommand Format:")
    print("1. Keyboard commands: '1,command_type,x,y,rotation,z'")
    print("2. Custom commands:  '2,Custom,y,x,rotation,z'")
    print("\nStarting test...")
    print("-" * 40)
    
    test_connection(HOST, PORTS) 