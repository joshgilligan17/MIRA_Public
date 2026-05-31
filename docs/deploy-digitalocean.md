# DigitalOcean Deployment

This is the current recommended deployment path for the class-project prototype:

- DigitalOcean Droplet for the Python/FastAPI backend and built React dashboard
- Docker Compose for repeatable deploys
- Caddy for HTTP/HTTPS reverse proxy
- MiniMax for report synthesis until Cloudflare Workers AI is approved

## 1. Local DigitalOcean Access

Install and authenticate `doctl` locally:

```bash
brew install doctl
export DIGITALOCEAN_ACCESS_TOKEN="dop_v1_..."
doctl auth init -t "$DIGITALOCEAN_ACCESS_TOKEN"
doctl account get
```

Do not commit or paste the token. Use a token with the deploy scopes described in the setup notes.

## 2. Create The Droplet

Recommended first size:

```text
Ubuntu 24.04 LTS
4 vCPU / 8 GB RAM
Region: sfo3 or nyc3
```

Create it in the DigitalOcean Control Panel or with `doctl`. Add your SSH key during creation.

Attach a Cloud Firewall allowing:

```text
TCP 22    from your IP if possible, otherwise 0.0.0.0/0 during setup
TCP 80    from 0.0.0.0/0
TCP 443   from 0.0.0.0/0
```

## 3. Bootstrap The Server

SSH into the Droplet:

```bash
ssh root@DROPLET_IP
```

Install Docker:

```bash
apt-get update
apt-get install -y ca-certificates curl git ufw
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
```

Prepare storage:

```bash
mkdir -p /opt/mira /data/mira/jobs /data/mira/projects
chown -R 10001:10001 /data/mira/jobs /data/mira/projects
```

## 4. Deploy MIRA

Clone the public repo:

```bash
git clone https://github.com/joshgilligan17/MIRA_Public.git /opt/mira
cd /opt/mira
```

Create server config:

```bash
cp .env.example .env
nano .env
```

For the first MiniMax-backed deployment, set:

```bash
MIRA_DOMAIN=:80
MIRA_DATA_DIR=/data/mira/jobs
MIRA_PROJECT_DIR=/data/mira/projects
MIRA_BASIC_AUTH_USERNAME=mira
MIRA_BASIC_AUTH_PASSWORD=use-a-long-random-password
MIRA_REPORT_PROVIDER=minimax
MIRA_REPORT_MODEL=MiniMax-M2.7
MIRA_REPORT_BASE_URL=https://api.minimax.io/v1
MIRA_REPORT_API_KEY=your_minimax_key
```

Start the app:

```bash
docker compose up -d --build
docker compose ps
curl http://127.0.0.1/api/health
```

Open:

```text
http://DROPLET_IP
```

The browser should prompt for the Basic Auth username/password.

## 5. Add A Domain

Point a DNS `A` record at the Droplet IP, then edit `.env`:

```bash
MIRA_DOMAIN=mira.example.com
```

Restart Caddy:

```bash
docker compose up -d
```

Caddy will request and renew TLS certificates automatically.

## 6. Updating

From the Droplet:

```bash
cd /opt/mira
git pull
docker compose up -d --build
docker image prune -f
```

## 7. Operational Notes

- Uploaded jobs persist under `/data/mira/jobs`; project folders, chat history, and target structures persist under `/data/mira/projects`.
- The default upload cap is `MIRA_MAX_UPLOAD_MB=250`.
- Keep Basic Auth enabled until Cloudflare Access is in front of the app.
- This deployment runs background jobs inside the web process, which is appropriate for class-project testing. A queue/worker split can come later.
- When Cloudflare Workers AI is approved, we can add a Cloudflare provider and switch synthesis without moving the backend.

## 8. Design-Model GPU Sessions

The default web Droplet should remain CPU-only. For real generative design, use short-lived GPU sessions:

```text
Local Apple Silicon: ProteinMPNN/LigandMPNN smoke tests and sequence design
DigitalOcean GPU: RFdiffusion backbones and BindCraft binder campaigns
MIRA CPU app: project storage, chat, analysis, filtering, synthesis
```

Recommended first GPU shape:

```text
RTX 6000 Ada or L40S, 48 GB VRAM
Expected use: RFdiffusion/BindCraft test campaigns
Budgeting: roughly $1.57/hour, so $250 covers about 150 GPU hours before storage/network overhead
```

Keep model weights on an attached volume or object storage cache, but destroy the GPU Droplet when idle. GPU Droplets can keep billing while powered off, so the safe operating pattern is:

```bash
# 1. Create GPU worker for a session.
# 2. Pull MIRA and model repos, attach/cache weights.
# 3. Set real backend env vars:
MIRA_RFDIFFUSION_REPO=/opt/models/RFdiffusion
MIRA_RFDIFFUSION_CONTIGS='[A1-100/0 80-120]'
MIRA_BINDCRAFT_REPO=/opt/models/BindCraft
MIRA_BINDCRAFT_SETTINGS=/opt/models/bindcraft/settings.json

# 4. Run the design worker/session.
# 5. Sync generated PDB/CIF/mmCIF outputs into the project.
# 6. Destroy the GPU worker after the campaign.
```

The current backend records design runs, generated structures, generated sequences, logs, and artifacts. The next production step is a separate GPU worker process that polls queued `rfdiffusion` and `bindcraft` design runs, writes outputs into the project design folder, then triggers the existing batch filtering path.
