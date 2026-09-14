# Deploy docextract on Vultr

Host the Gradio PDF pipeline on a Vultr Cloud Compute instance.

## Recommended Vultr plan

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| vCPU | 2 | 4 |
| Disk | 50 GB | 80 GB |
| OS | Ubuntu 22.04 or 24.04 LTS | same |

First boot downloads ~500 MB of ONNX models. Extraction uses CPU heavily (layout + OCR + tables).

## 1. Create the server

1. [Vultr](https://www.vultr.com/) → **Deploy** → **Cloud Compute**.
2. Choose a region close to your users.
3. Image: **Ubuntu 24.04 LTS**.
4. Plan: at least **4 GB RAM / 2 vCPU**.
5. Add your SSH key.
6. Deploy and note the **public IP**.

## 2. Bootstrap Docker (on the VPS)

```bash
ssh root@YOUR_VULTR_IP

git clone https://github.com/YOUR_USER/docextract.git /opt/docextract
cd /opt/docextract
bash deploy/vultr/setup.sh
```

Or upload the project with `scp` instead of `git clone`.

## 3. Start the app

```bash
cd /opt/docextract
docker compose up -d --build
docker compose logs -f
```

Wait until you see `Running on local URL: http://0.0.0.0:7860` and model download completes.

**Quick test (on the server):**

```bash
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860
```

Should return `200`.

## 4. HTTPS with a domain (recommended)

Point a DNS **A record** to your Vultr IP, e.g. `extract.example.com`.

```bash
apt-get install -y nginx certbot python3-certbot-nginx

cp /opt/docextract/deploy/vultr/nginx-docextract.conf /etc/nginx/sites-available/docextract
sed -i 's/YOUR_DOMAIN/extract.example.com/g' /etc/nginx/sites-available/docextract
ln -sf /etc/nginx/sites-available/docextract /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

certbot --nginx -d extract.example.com
```

Open `https://extract.example.com` in a browser.

## 5. Updates

```bash
cd /opt/docextract
git pull
docker compose up -d --build
```

Model cache persists in the Docker volume `hf_cache`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM / container restarts | Upgrade to 8 GB RAM or set `max_workers=1` in API calls |
| Slow first start | Normal — models download once into `hf_cache` |
| 502 from nginx | `docker compose ps` — wait for app to finish model prep |
| Large PDF upload fails | Increase `client_max_body_size` in nginx config |

## Optional: expose port without nginx (dev only)

In `docker-compose.yml`, change ports to:

```yaml
ports:
  - "7860:7860"
```

Then open `http://YOUR_VULTR_IP:7860` and allow port 7860 in Vultr firewall + `ufw`. **Use HTTPS in production.**
