#!/bin/sh
# Runs scripts/<role>.py for "learner" or "worker"; any other command runs as given.
case "$1" in
    learner|worker) role=$1; shift; exec python /opt/rl-split/scripts/"$role".py "$@" ;;
    *) exec "$@" ;;
esac
