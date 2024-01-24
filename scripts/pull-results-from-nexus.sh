#!/bin/zsh

# This is the shell script that we used to recover results (analysis and
# checkpoints) from UMD's Nexus cluster. If you're attempting to look through
# our work, this file is likely not relevant to you.

source .env

REQUESTED_DIRECTORIES="analysis"

REMOTE_ARTIFACTS=$(ssh nexus "zsh" <<-SCRIPT
    cd $NEXUS_REMOTE
    source .env
    echo \$ARTIFACTS
SCRIPT
)

REMOTE_FILESIZE=$(ssh nexus "zsh" <<-SCRIPT
    cd $REMOTE_ARTIFACTS
    du -bc $REQUESTED_DIRECTORIES | grep total | python3 -c 'print(input().split()[0])'
SCRIPT
)

cd artifacts

(ssh nexus "zsh" <<-SCRIPT
    cd $REMOTE_ARTIFACTS
    tar cf - $REQUESTED_DIRECTORIES | lz4 -
SCRIPT
) | unlz4 - | pv -s $REMOTE_FILESIZE | gtar xf -