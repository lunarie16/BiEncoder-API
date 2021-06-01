#!/usr/bin/env bash

IMAGE=registry.datexis.com/mmenke/predict-ner-nel

version=0.1.7
echo "Version: $version"
docker build -t $IMAGE -t $IMAGE:$version ../.
docker push $IMAGE:$version
echo "Done pushing image $image for build $version"