#!/usr/bin/env bash
set -euo pipefail

kubectl apply -f 00-task-git-clone.yaml
kubectl apply -f 01-task-harbor-login.yaml
kubectl apply -f 02-task-kaniko-build-push.yaml
kubectl apply -f 03-pipeline.yaml
kubectl apply -f 04-pipelinerun.yaml
kubectl apply -f harbor-secret.yaml