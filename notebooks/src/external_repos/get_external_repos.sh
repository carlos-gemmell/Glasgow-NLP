#!/bin/bash

echo "Getting TranX"
git clone https://github.com/pcyin/tranX.git

echo "Getting tree-sitter-python"
git clone https://github.com/tree-sitter/tree-sitter-python

echo "Getting anserini"
git clone --recurse-submodules https://github.com/castorini/anserini.git
cd anserini
cd eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make
mvn clean package appassembler:assemble -DskipTests -Dmaven.javadoc.skip=true

echo "Getting GRILL CAsT tools"
git clone https://github.com/grill-lab/trec-cast-tools.git