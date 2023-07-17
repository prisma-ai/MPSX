import subprocess

IGNORED_FUNCS = [
    "`if`",
    "`while`",
    "`for`",
    "`for`",
    "placeholder",
    "variable",
    "assign",
    "adam",
]


def call_subprocess(cmd):
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    return stdout.decode('utf-8')


def get_mpsgraph_swift_interface():
    cmd = """echo "import MetalPerformanceShadersGraph\n:type lookup MPSGraph" | swift repl"""
    return call_subprocess(cmd)


def parse_func(signature):
    cmd = "sourcekitten structure --text '${signature}'"
    return call_subprocess(cmd)


def parse_func(parts):
    params = []
    outs = []

    func_started = False
    func_ended = False

    for part in parts:
        if part == "obsoleted:":
            return None

        if part == "func":
            func_started = True
            continue

        if part == "->":
            func_ended = True
            continue

        if func_started:
            if func_ended:
                outs.append(part)
            else:
                params.append(part)

    if len(params) == 0 or len(outs) != 1:
        return None

    if "name:" not in params:
        return None

    func_name, first_param_name = params[0].split('(')

    if "gradient" in func_name.lower() or func_name in IGNORED_FUNCS:
        return None

    # params[0] = first_param_name

    # params = [p if p == '_' else p[:-1].split('.')[0] for p in params]
    signature = ' '.join(params + ['->'] + outs)

    return signature


for line in get_mpsgraph_swift_interface().splitlines():
    parts = [s for s in line.split(' ') if s]
    if "func" in parts:
        # for part in parts:
        # if "split" in part:
        # print(line)
        # print(line)
        signature = parse_func(parts)

        if signature:
            print(signature)
            # print(parse_func(signature))
    # elif "@available(macOS" in parts:
        # print(parts)
