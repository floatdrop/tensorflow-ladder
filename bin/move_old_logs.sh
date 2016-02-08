#!/usr/bin/env bash

mkdir -p old_logs
find logs -type d -maxdepth 1 -mmin +10 -exec mv '{}' old_logs \;
