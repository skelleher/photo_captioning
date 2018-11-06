#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder of images to caption>"
    exit -1
fi

for filename in $1/*; do
    echo $filename
    curl --silent -X POST http://hiro_wifi:32884/query -H "Content-type: application/octet-stream" --data-binary @$filename
done

