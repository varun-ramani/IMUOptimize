#!/bin/zsh

# This is the shell script that we used to deploy our latest scripts and source
# code to UMD's Nexus cluster. If you're attempting to look through our work,
# this file is likely not relevant to you.

source .env

$LOCAL_TAR_COMMAND cf - scripts src \
    | pv \
    | lz4 - \
    | ssh nexus "cd $NEXUS_REMOTE; unlz4 - | tar xf -"