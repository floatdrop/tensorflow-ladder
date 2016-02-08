#!/usr/bin/env bash

AGE_IN_MINUTES=${1-10} # first parameter or ten minutes by default

mkdir -p old_logs
find logs -type d -mindepth 1 -maxdepth 1 -mmin +$AGE_IN_MINUTES -exec mv '{}' old_logs \;
