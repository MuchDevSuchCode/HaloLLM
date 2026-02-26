#!/bin/bash

HOST="http://halo.lan:11435"
PROMPT=${1:-"Hello from the Strix Halo command line!"}

echo "=========================================="
echo " 🚀 Pinging HaloLLM Backend"
echo "=========================================="
echo "Target : $HOST/api/generate"
echo "Prompt : \"$PROMPT\""
echo "------------------------------------------"

# We use curl with the -w flag to measure the exact total round-trip time
curl -s -w "\n------------------------------------------\n⏱️  Total HTTP Latency: %{time_total} seconds\n" \
     -X POST $HOST/api/generate \
     -H "Content-Type: application/json" \
     -d "{
           \"prompt\": \"$PROMPT\",
           \"model_path\": \"/home/ai/HaloLLM/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\"
         }"

echo "=========================================="