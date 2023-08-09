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


def parse_swift_func(signature):
    cmd = f'sourcekitten structure --text \'{signature}\''
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
    
    # params.pop(0)
    
    # if first_param_name != '_':
        # func_name = func_name + first_param_name.capitalize()
    # print(func_name)

    # params = list(filter(lambda param: param[-1] in (',', ')', ':'), params))
    # params = list(map(lambda param: param[:-1], params))
    # params = list(zip(params[::2], params[1::2]))
    # params = list(map(lambda param: (param[0], param[1].split('.')[1]), params))

    # if params[0][1] == 'MPSGraphTensor':
    #     params.pop(0)

    # if params[-1][0] == 'name':
    #     params.pop(-1)

    # for param in params:
        # if param[-1] not in (',', ')', ':'):
        #     continue
        # print(param)
    # print(params)

    # params[0] = first_param_name

    # params = [p if p == '_' else p[:-1].split('.')[0] for p in params]
    # signature = 'func ' + ' '.join(params + ['->'] + outs)

    # return signature

# print(get_mpsgraph_swift_interface())

# text_file = open("sample.json", "w")
# n = text_file.write(parse_swift_func(get_mpsgraph_swift_interface()))
# text_file.close()

# for line in get_mpsgraph_swift_interface().splitlines():
#     print(parse_swift_func(line))
#     parts = [s for s in line.split(' ') if s]
#     if "func" in parts:
#         # for part in parts:
#         # if "split" in part:
#         # print(line)
#         # print(line)
#         signature = parse_func(parts)

#         if signature:
#             print(signature)
#             print(parse_swift_func(signature))
#     elif "@available(macOS" in parts:
#         print(parts)


