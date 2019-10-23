#!/usr/bin/env python3
import argparse
import json
import ssl
import urllib.request
 
TOKEN='eyJhbGciOiJSUzI1NiIsImtpZCI6IiJ9.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJvcGVuc2hpZnQtaW5mcmEiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoibW9uaXRvcmluZy10b2tlbi1send2eiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJtb25pdG9yaW5nIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiZmZhMjBjZmQtZWE4Yi0xMWU5LWE4ZjgtYWMxZjZiYmI2MzU0Iiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50Om9wZW5zaGlmdC1pbmZyYTptb25pdG9yaW5nIn0.huN0fDFWqfrm7pjnwQ-bRXh9XLTzTSnyGEW4AfqhyvLKXiUYuvRQSyNCvfL53bMRs_6QKzg55XE1SdSfF0rHzi78PObpG7AFPp0NZ637f-ZkyyOCgUbAjqdEMJ-qZz0RNx_qHKzuHTa5ECqqJ8GwiyOZ7WpkGubVSY5bA9nPwf0TdW4SkJIyBFqwOWh23IQiN-CsaXyxA7Eeic4gftVx4LLXXcloYnMBzg9n39LFH0Et4glrgbYI8ypxCiv2kYl3XUtB4MKtNJo5tBWGwvkvEmNOjNONC5pLRRtU9fA6BLEWrbS0E-sPTGMp_CbP2SNSTCrBvlalSKK72DEoX50TeQ'

ENDPOINT='https://idagpu-head.dcs.gla.ac.uk:8443'

NODES = ['idagpu-01.dcs.gla.ac.uk', 'idagpu-02.dcs.gla.ac.uk']

ONE_GIB = 1024 * 1024 * 1024
ONE_MIB = 1024 * 1024

def cpu_to_shares(val):
    if not val.endswith('m'):
        return int(val) * 1000

    return int(val[:-1])

def mem_to_bytes(val):
    lookup = {
              'Ki': 1024,
              'Mi': ONE_MIB,
              'Gi': ONE_GIB,
              'K' : 1000,
              'M' : 1000 * 1000,
              'G' : 1000 * 1000 * 1000,
             }

    if val[-1].isdigit():
        return int(val)

    for suffix, divisor in lookup.items():
        if val.endswith(suffix):
            val = int(val[:-len(suffix)])
            return val * divisor

    raise Exception('Unknown suffix for mem value: {}'.format(val))

def get_pods_on_node(node, endpoint, token):
    request = urllib.request.Request(endpoint + '/api/v1/pods?fieldSelector=spec.nodeName={},status.phase!=Failed,status.phase!=Succeeded'.format(node))
    request.add_header('Authorization', 'Bearer ' + token)
    request.add_header('Accept', 'application/json')
    ssl_context = ssl._create_unverified_context() # required for self-signed cert
    result = urllib.request.urlopen(request, context=ssl_context)
    return result.read()

def collect_pod_data(data):
    data = json.loads(data.decode('utf-8'))
    items = data['items']
    pod_data = {'total_cpu_lim': 0, 'total_cpu_req': 0, 'total_mem_req': 0, 'total_mem_lim': 0, 'total_gpu_req': 0, 'total_gpu_lim': 0}
    pod_data['names'] = []
    pod_data['running'] = 0
    pod_data['pending'] = 0
    for item in items:
        metadata = item['metadata']
        containers = item['spec']['containers']
        status = item['status']

        pod_data['names'].append(metadata['name'])
        if status['phase'] == 'Running':
            pod_data['running'] += 1
        elif status['phase'] == 'Pending':
            pod_data['pending'] += 1
        else:
            raise Exception('Unexpected phase: {}'.format(status['phase']))

        for c in containers:
            resources = c['resources']

            limits = resources.get('limits', {})
            requests = resources.get('requests', {})
            pod_data['total_cpu_req'] += cpu_to_shares(requests.get('cpu', '0'))
            pod_data['total_cpu_lim'] += cpu_to_shares(limits.get('cpu', '0'))
            pod_data['total_mem_req'] += mem_to_bytes(requests.get('memory', '0'))
            pod_data['total_mem_lim'] += mem_to_bytes(limits.get('memory', '0'))
            pod_data['total_gpu_req'] += int(requests.get('nvidia.com/gpu', '0'))
            pod_data['total_gpu_lim'] += int(limits.get('nvidia.com/gpu', '0'))
    return pod_data

def get_node_info(name, endpoint, token):
    request = urllib.request.Request(endpoint + '/api/v1/nodes/{}'.format(name))
    request.add_header('Authorization', 'Bearer ' + token)
    request.add_header('Accept', 'application/json')
    ssl_context = ssl._create_unverified_context() # required for self-signed cert
    result = urllib.request.urlopen(request, context=ssl_context)
    return result.read()

def display_node(name, endpoint, token):
    node_info = json.loads(get_node_info(name, endpoint, token))
    labels = node_info['metadata']['labels'] # (includes kubernetes node types etc, but also the 2080ti/titanrtx labels)
    status = node_info['status'] # most of the interesting stuff here
    node_resources = {'capacity': status['capacity'], 'allocatable': status['allocatable']}
    conditions = status['conditions']
    state = 'OK'
    for cond in conditions:
        if cond['status'] != "False" and cond['type'] != "Ready":
            state = 'FAIL'
        elif cond['status'] != "True" and cond['type'] == "Ready":
            state = 'FAIL'

    gpu_type = 'RTX 2080Ti' if 'node-role.ida/gpu2080ti' in labels else 'Titan RTX'

    pod_data = collect_pod_data(get_pods_on_node(name, endpoint, token))
    
    print('[{}]'.format(name))
    print('  State    : {}'.format(state))
    print('  Pods     : running={}, pending={}'.format(pod_data['running'], pod_data['pending']))
    print('  CPU cores: {}'.format(node_resources['allocatable']['cpu']))
    print('  RAM      : {:d}Gi'.format(int(mem_to_bytes(node_resources['allocatable']['memory']) / (ONE_GIB))))
    print('  GPUs     : {}x {}'.format(node_resources['allocatable']['nvidia.com/gpu'], gpu_type))
    print('  Resources:')
    cpu_req, cpu_lim = pod_data['total_cpu_req'], pod_data['total_cpu_lim']
    cpu_per_req = int(100 * cpu_req / cpu_to_shares(node_resources['allocatable']['cpu']))
    cpu_per_lim = int(100 * cpu_lim / cpu_to_shares(node_resources['allocatable']['cpu']))
    print('    CPU requests: {}m ({:d}%)'.format(cpu_req, cpu_per_req))
    print('    CPU limits  : {}m ({:d}%)'.format(cpu_lim, cpu_per_lim))
    mem_req, mem_lim = pod_data['total_mem_req'], pod_data['total_mem_lim']
    mem_per_req = int(100 * mem_req / mem_to_bytes(node_resources['allocatable']['memory']))
    mem_per_lim = int(100 * mem_lim / mem_to_bytes(node_resources['allocatable']['memory']))
    print('    RAM requests: {:.0f}Gi ({:d}%)'.format(mem_req / (ONE_GIB), mem_per_req))
    print('    RAM limits  : {:.0f}Gi ({:d}%)'.format(mem_lim / (ONE_GIB), mem_per_lim))
    gpu_req, gpu_lim = pod_data['total_gpu_req'], pod_data['total_gpu_lim']
    print('    GPU requests: {}/{}'.format(gpu_req, node_resources['allocatable']['nvidia.com/gpu']))
    print('    GPU limits  : {}/{}'.format(gpu_lim, node_resources['allocatable']['nvidia.com/gpu']))
    print('')

def main():
    for n in NODES:
        display_node(n, ENDPOINT, TOKEN)

if __name__ == "__main__":
    main()
