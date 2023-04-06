#!/bin/bash

#echo "### check TFaaS models"
#curl http://localhost:8083/models
#sleep 3

echo
echo "### upload mnist ML trained tarball"
tar tvfz ./mnist.tar.gz
sleep 3

echo
echo "### upload mnist ML trained tarball"
curl -v -s -X POST -H "Content-Encoding: gzip" -H "content-type: application/octet-stream" --data-binary @./mnist.tar.gz http://localhost:8083/upload
sleep 3

echo
echo "### check if out model exist in TFaaS server"
curl http://localhost:8083/models | jq '.[] | select(.name == "mnist")'
sleep 3

echo
echo "### identify our image, img1.png (number 1)"
curl http://localhost:8083/predict/image -F 'image=@./img1.png' -F 'model=mnist'
sleep 3

echo
echo "### identify our image, img4.png (number 4)"
curl http://localhost:8083/predict/image -F 'image=@./img4.png' -F 'model=mnist'
