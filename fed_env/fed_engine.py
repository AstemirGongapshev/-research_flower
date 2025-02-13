import asyncio
import os

SERVER_SCRIPT = "./fed_env/server.py"
CLIENT_SCRIPTS = ["./fed_env/client_1_torch.py", "./fed_env/client_2_torch.py"]  # Список клиентских скриптов

async def start_server():

    print("Starting server...")
    env = os.environ.copy()
    process = await asyncio.create_subprocess_exec(
        "python", SERVER_SCRIPT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    return process


async def start_client(script_name: str, client_id: int):
    print(f"Starting client {client_id} ({script_name})...")
    env = os.environ.copy()
    env["CLIENT_ID"] = str(client_id)
    process = await asyncio.create_subprocess_exec(
        "python", script_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )
    return process


async def read_stream(stream, name):
    while True:
        line = await stream.readline()
        if not line:
            break
        try:
            decoded_line = line.decode('utf-8').strip()
        except UnicodeDecodeError:
            decoded_line = line.decode('cp1251', errors='replace').strip()
        print(f"[{name}] {decoded_line}")


async def run_process_with_output(process, name):
    await asyncio.gather(
        read_stream(process.stdout, name),
        read_stream(process.stderr, name),
    )
    await process.wait()


async def main():
    # Start the server
    server = await start_server()
    asyncio.create_task(run_process_with_output(server, "Server"))

    await asyncio.sleep(2)  # Allow the server to initialize

    # Start clients
    clients = []
    for i, script_name in enumerate(CLIENT_SCRIPTS, start=1):
        client = await start_client(script_name, client_id=i)
        clients.append(client)
        asyncio.create_task(run_process_with_output(client, f"Client {i}"))

    # Wait for all clients and the server to finish
    try:
        await asyncio.gather(
            server.wait(),
            *[client.wait() for client in clients]
        )
    except KeyboardInterrupt:
        print("Terminating all processes...")
        for client in clients:
            client.terminate()
        server.terminate()


if __name__ == "__main__":
    asyncio.run(main())
