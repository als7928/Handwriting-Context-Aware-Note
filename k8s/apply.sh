#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f 01-configmaps.yaml
kubectl apply -f 02-secret.yaml
kubectl apply -f 03-pvcs.yaml
kubectl apply -f 11-qdrant.yaml
kubectl apply -f 12-backend.yaml
kubectl apply -f 13-frontend.yaml
kubectl apply -f 20-ingress.yaml