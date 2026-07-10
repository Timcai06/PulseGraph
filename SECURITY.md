# PulseGraph Security Boundary

PulseGraph currently supports one deployment model: a single user running the frontend and backend on the same machine.

## Trusted Python Execution

Training resources and attached model source are executable Python. Resource preview, training, source attachment, and forward replay may import modules and run their top-level code. Only load code you wrote or reviewed.

The current safeguards are defense in depth for this local workflow:

- the provided server commands bind to `127.0.0.1`;
- the API rejects non-loopback clients and non-local browser origins;
- Python execution routes require the explicit `X-PulseGraph-Trust: trusted-local-code` marker;
- weights-only artifact inspection uses restricted PyTorch loading;
- live-run memory, subscriber queues, and event persistence have bounded or locked write paths.

The trust marker is not authentication, and a local subprocess is not a security sandbox. PulseGraph does not currently isolate uploaded Python from the host filesystem, network, environment variables, or the permissions of the user running the backend.

## Unsupported Deployments

Do not expose the current backend to a LAN, public network, reverse proxy, shared workstation, or multi-user environment. CORS is not an authorization boundary, and runs do not have per-user ownership.

Before a shared deployment is supported, PulseGraph needs authenticated users, run ownership checks, isolated worker processes or virtual machines, resource quotas, and a durable transactional event store.
