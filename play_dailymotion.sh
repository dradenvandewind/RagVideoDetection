#!/bin/bash

ID=$1

URL=$(curl -s "https://www.dailymotion.com/player/metadata/video/$ID" \
 | jq -r '.qualities.auto[0].url')

ffmpeg -i "$URL" -c copy "$ID.mp4"
