# Shell functions for running Julia scripts
# Local-first execution wrapper

Julia_pc_and_local() {
    if [ "$1" == "--no-job-name" ]; then
        shift
    fi

    print_info Julia "$@"
    julia "$@"
}

print_info() {
    software=$1
    shift
    if [ $# -eq 1 ]; then
        echo "Running ${1} via ${software}, waiting..."
    else
        echo "Running ${1} via ${software} with args = ${*:2}, waiting..."
    fi
}
