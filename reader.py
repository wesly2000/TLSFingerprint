import pandas as pd
import warnings
import json
try:
    from .algorithm import injection_tls
except:
    from algorithm import injection_tls

class Reader(object):
    """
    Reader object for extracting features from .pcap files
    Params:
        verbose : boolean, default=False
            Boolean indicating whether to be verbose in reading
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def read(self, path):
        """
        A wrapper function of read_json
        Args:
            path: string
                The .json file path. 

        Warning:
            Method throws warning if tshark is not available.
        """

        # if verbose, print the file being reading
        if self.verbose:
            print("Reading file {}".format(path))

        # test if tshark is available, if not, try slower pyshark
        try:
            return self.read_json(path)
        except Exception as ex:
            warnings.warn("json loader error: '{}'"
                          .format(ex))
            raise NotImplementedError


    def read_json(self, path, version='3.3.6'):
        """
        Fetch TLS packets from .json files, and extract given features
        from each packet.
        Args:
            path : string
                Path to .pcap file to read.
            version: string
                TShark version used for network collection.
        Returns:
            result : pd.DataFrame=(n_packets, n_features)
                Where features consist of:
                0) Filename of capture
                1) Timestamp
                2) TCP stream identifier
                3) IP packet source
                4) IP packet destination
                5) tls.record.length
                6) tls.record.content_type
                7) tls.handshake.type        
        """
        # Set certificate command based on version
        # tshark versions <3 use ssl.handshake.certificate
        # tshark versions 3+ use tls.handshake.certificate
        certificate  = "ssl" if int(version.split('.')[0]) < 3 else "tls"
        certificate += ".handshake.certificate"
        result = []

        _dicts = Reader.load(path)
        for _dict in _dicts:
            if 'tls' not in _dict['_source']['layers']: continue
            _layers = _dict['_source']['layers']
            timestamp = _layers['frame']['frame.time_epoch']
            tcp_stream = _layers['tcp']['tcp.stream']
            ip_src = _layers['ip']['ip.src']
            ip_dst = _layers['ip']['ip.dst']
            #BUG: 2023/3/9-15:34,In single packet, there might be multiple tls layer
            #FIX: See function json_resolver
            #BUG: 2023/3/10-14:46, _layers['tls'] might be dict as well as list of dicts
            #FIX: Decide the type of _layers['tls']
            tls_layers = [_layers['tls']] if isinstance(_layers['tls'], dict) else _layers['tls']
            for tls_layer in tls_layers:
                # BUG: 2023/1/10-12:31, after tsharking, some TLS Layer data
                # contains nothing but ""tls": "Transport Layer Security""
                # reasons still seeking. 
                # FIX: A awkward fix is to skip following
                # processing as long as @record is a string.
                if isinstance(tls_layer, str): continue
                #BUG: 2023/3/10-14:55, tls_layer['tls.record'] might be dict as well as list of dicts
                #FIX: Decide the type of tls_layer['tls.record']
                try:
                    _ = tls_layer['tls.record']
                except:
                    continue
                records = [tls_layer['tls.record']] if isinstance(tls_layer['tls.record'], dict) else tls_layer['tls.record']
                for record in records:
                    length = record['tls.record.length']
                    # BUG: 2023/2/15-23:26 KeyError: 'tls.record.content_type'
                    # FIX: This code is for new tls content_type/handshake_type detection.
                    try:
                        content_type = record['tls.record.content_type']
                    except:
                        # print(
                        #     "Exception occurs in file {}\nTCP Stream: {}, Time Stamp {}".format(
                        #         path, tcp_stream, timestamp
                        #     )
                        # )
                        continue
                    if content_type not in ['20','21','22','23']: continue
                    try:
                        handshake_type = record['tls.handshake']['tls.handshake.type']
                    except:
                        handshake_type = None
                        
                    packet = [path, timestamp, tcp_stream, ip_src, ip_dst, length, content_type, handshake_type]
                    result.append(packet)
        result = pd.DataFrame(
            data=result,
            columns=['path', 'timestamp', 'tcp_stream', 'ip_src', 'ip_dst', 'length', 'content_type', 'handshake_type']
        )
        return result
    
    
    @staticmethod
    def load(path):
        json_resolver = JSONResolver()
        with open(path, 'rb') as f:
            data = json.load(fp=f, object_pairs_hook=json_resolver)
        return data        


class JSONResolver(object):
    def __init__(self):
        pass

    def __call__(self, pairs):
        return JSONResolver.json_resolver(pairs)

    @staticmethod
    def json_resolver(ordered_pairs):
        '''
        In a single TLS packet, there might be several tls.record.handshake,
        which corresponds to several Encrypted Data Layer, etc.
        Therefore, distinguishing them is nessesary:
        json_resolver convert repeatitive keys in .json file into a key-value_list
        format. E.g., in example.json:
        [
            {
                a: b,
                a: c,
            }
        ]
        In Python dictionary, it is forbidden to have repeatitive keys like:
        {a:b, a:c} 
        Therefore, json_resolver transforms it into:
        {a:[b,c]} 
        '''
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                if type(d[k]) == list:
                    d[k].append(v)
                else:
                    newlist = []
                    newlist.append(d[k])
                    newlist.append(v)
                    d[k] = newlist
            else:
                d[k] = v
        return d