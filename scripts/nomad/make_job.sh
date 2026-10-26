#!/usr/bin/env bash
# Generate scripts/nomad/pong-reinforce.nomad: a Nomad batch job that trains 5 seeds x {none, value}
# of Pong REINFORCE on mlds-deepdish5 (CPU only), evaluates 500 episodes per run and packs the results.
# The code is embedded as a base64 tarball in a template block (the cluster has no shared storage).
set -e
cd "$(dirname "$0")/../.."
REV=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
HOURS=${HOURS:-4.0}
tar -czf /tmp/pong_code.tar.gz policy_gradient/*.py shuttle_mdp/*.py tests/*.py viz.py requirements.txt
B64=$(base64 < /tmp/pong_code.tar.gz)
cat > scripts/nomad/pong-reinforce.nomad <<EOF
job "pong-reinforce" {
  datacenters = ["dc1"]
  type        = "batch"

  meta {
    revision = "$REV"
    hours    = "$HOURS"
  }

  constraint {
    attribute = "\${attr.unique.hostname}"
    value     = "mlds-deepdish5"
  }

  group "main" {
    count = 1

    restart {
      attempts = 0
      mode     = "fail"
    }

    reschedule {
      attempts  = 0
      unlimited = false
    }

    ephemeral_disk {
      sticky  = true
      migrate = false
      size    = 10000
    }

    task "run" {
      driver = "docker"

      config {
        image   = "python:3.12-slim"
        command = "/bin/sh"
        args    = ["/local/run.sh"]
      }

      env {
        NODE_NAME            = "\${node.unique.name}"
        OMP_NUM_THREADS      = "1"
        MKL_NUM_THREADS      = "1"
        OPENBLAS_NUM_THREADS = "1"
        PYTHONUNBUFFERED     = "1"
        HOURS                = "$HOURS"
      }

      template {
        destination = "local/run.sh"
        data        = <<PAYLOAD_EOF
$(cat scripts/nomad/run.sh)
PAYLOAD_EOF
      }

      template {
        destination = "local/code.b64"
        data        = <<PAYLOAD_EOF
$B64
PAYLOAD_EOF
      }

      resources {
        cores  = 24
        memory = 32768
      }
    }
  }
}
EOF
echo "wrote scripts/nomad/pong-reinforce.nomad (revision $REV, $(wc -c < scripts/nomad/pong-reinforce.nomad) bytes)"
