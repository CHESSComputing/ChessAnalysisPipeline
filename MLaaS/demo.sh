#!/bin/bash
# TFaaS host name
turl=http://localhost:8083
# location of files
tdir=/home/vk/chess/TFaaS/models
# we need the following files
model_tarball=$tdir/model.tar.gz
params_json=$tdir/vk/params.json
input_json=$tdir/vk/input.json
upload_json=$tdir/vk/upload.json
model_pb=$tdir/vk/model.pb
labels_txt=$tdir/vk/labels.txt
tfaas_client=$tdir/tfaas_client.py
hey=$tdir/hey-tools/hey_amd64

echo "### obtain any existing ML model"
echo
echo "$tfaas_client --url=$turl --models"
echo
$tfaas_client --url=$turl --models
sleep 1
echo

echo "### upload new ML model"
echo
echo "cat $upload_json"
echo
cat $upload_json
echo
echo "$tfaas_client --url=$turl --upload=$upload_json"
$tfaas_client --url=$turl --upload=$upload_json
echo

echo "### view if our model exists"
echo
echo "$tfaas_client --url=$turl --models"
echo
$tfaas_client --url=$turl --models
sleep 2
echo

echo "### view if our model exists, but use jq tool to get better view over JSON"
echo
echo "$tfaas_client --url=$turl --models | jq"
echo
$tfaas_client --url=$turl --models | jq
sleep 2
echo

echo "### let's obtain some prediction"
echo
echo "cat $input_json"
echo
cat $input_json
echo
echo "$tfaas_client --url=$turl --predict=$input_json"
echo
$tfaas_client --url=$turl --predict=$input_json
sleep 2
echo

echo "### let's delete our ML model named vk"
echo
echo "$tfaas_client --url=$turl --delete=vk"
echo
$tfaas_client --url=$turl --delete=vk
sleep 1
echo

echo "### lets view again available models"
echo
echo "$tfaas_client --url=$turl --models"
echo
$tfaas_client --url=$turl --models
sleep 2
echo

echo "### now let's use curl as CLI tool to communicate with TFaaS"
echo
sleep 5

echo "### Let's view our models"
echo
echo "curl -s $turl/models"
echo
curl -s $turl/models
sleep 1
echo

echo "### let's send POST HTTP request with our parameters to upload ML model"
echo "### we provide $params_json"
echo
cat $params_json
echo
echo "### we provide $model_pb TF model"
echo 
ls -al $model_pb
echo
echo "### and we provide our labels in $labels_txt file"
echo
cat $labels_txt
echo
echo "### now we make curl call"
echo
echo "curl -s -X POST $turl/upload -F 'name=vk' -F 'params=@$params_json' -F 'model=@$model_pb' -F 'labels=@$labels_txt'"
echo
curl -s -X POST $turl/upload -F 'name=vk' -F "params=@$params_json" -F "model=@$model_pb" -F "labels=@$labels_txt"
sleep 1
echo

echo "### Now we can view our models"
echo
echo "curl -s $turl/models | jq"
echo
curl -s $turl/models | jq
echo
sleep 2

echo "### And we can obtain our predictions using /json API"
echo
echo "curl -s -X POST $turl/json -H "Content-type: application/json" -d@$input_json"
echo
curl -s -X POST $turl/json -H "Content-type: application/json" -d@$input_json
sleep 1
echo

echo "### Now we can delete ML model using /delete end-point"
echo
echo "curl -s -X DELETE $turl/delete -F 'model=vk'"
echo
curl -s -X DELETE $turl/delete -F 'model=vk'
sleep 1
echo

echo "### Now we can view our models"
echo
echo "curl -s $turl/models"
echo
curl -s $turl/models
echo
sleep 1

$tfaas_client --url=$turl --upload=$upload_json

echo
echo "### now let's use tar ball and upload it"
echo
ls -al $model_tarball
tar tvfz $model_tarball
sleep 5

echo "curl -v -X POST -H \"Content-Encoding: gzip\" -H \"content-type: application/octet-stream\" --data-binary @$model_tarball $turl/upload"
curl -v -X POST -H"Content-Encoding: gzip" -H"content-type: application/octet-stream" --data-binary @$model_tarball $turl/upload
sleep 1
echo

echo "### Now we can view our models"
echo
echo "curl -s $turl/models | jq"
echo
curl -s $turl/models | jq
echo
sleep 2

echo "### And we can obtain our predictions using /json API"
echo
echo "curl -s -X POST $turl/json -H "Content-type: application/json" -d@$input_json"
echo
curl -s -X POST $turl/json -H "Content-type: application/json" -d@$input_json
sleep 1
echo

if [ -f $hey ]; then
echo "### Now let's perform some stress tests"
echo "### for that we'll use hey tool which will send number of concurrent requests to tfaas service"
echo
echo "$hey -m POST -H "Content-type: application/json" -D $input_json $turl/json"
$hey -m POST -H "Content-type: application/json" -D $input_json $turl/json
fi
