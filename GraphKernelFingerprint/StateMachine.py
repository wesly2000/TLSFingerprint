import numpy as np
import pandas as pd
import networkx as nx
from .reader import Reader
from matplotlib import pyplot as plt
from .algorithm import injection_tls, cantor_index, angle_hash

STATE_NUM = 289
# The attributes of vertices extracted from TLS packets.
ATTR = ['length']

class Session():
    def __init__(self, packets: pd.DataFrame, label):
        """ 
        A group of packets with the same (filename, stream)
        Params:
            len: int,
                The number of packets within a session.
            identifier: tuple (filename: str, stream: int)
                The UNIQUE id identifying a session.

        Args:
            packets: pd.DataFrame,
                The packets within a session.
        """
        self.len = packets.shape[0]
        assert packets['path'].drop_duplicates().shape[0] == 1, "There are more than 1 files for generating a Session."
        assert packets['tcp_stream'].drop_duplicates().shape[0] == 1, "There are more than 1 streams for generating a Session."
        self._identifier = (packets.iloc[0, 0], packets.iloc[0, 2])
        self._packets = packets
        self._label = label

    def first_order_transition(self):
        '''
        List of packet type transition sequence.
        '''
        seq = []
        for _, row in self.packets.iterrows():
            seq.append(injection_tls((row['content_type'], row['handshake_type'])))
        return seq
    
    def length_transition(self):
        '''
        List of packet length transition sequence.
        '''
        seq = []
        for _, row in self.packets.iterrows():
            seq.append(int(row['length']))
        return seq
    
    def second_order_transition(self):
        '''
        List of packet type second order transition sequence. 
        '''
        before_index = 0
        after_index = 0
        seq = [before_index]
        pair = [0, 0] # (Empty, Empty)
        for idx, row in self.packets.iterrows():
            content_type = row['content_type']
            handshake_type = row['handshake_type']
            # The TLS packet state.
            try:
                state = injection_tls((content_type, handshake_type))
            except:
                print("Exception occurs when handling session {}".format(self.identifier),
                    "At packet {}".format(idx))
                raise TypeError
            pair[0] = pair[1]
            pair[1] = state
            before_index = after_index
            after_index = cantor_index(pair[0], pair[1])
            seq.append(after_index)
        return seq


    @property
    def identifier(self):
        return self._identifier
    
    @property
    def packets(self):
        return self._packets
    
    @property
    def label(self):
        return self._label

    def __len__(self):
        return self.len

def sessionGenerate(path, label=0):
    ''' Generate sessions from a .json
    Args:
        path: str,
            file path to .json
        label: int,
            the class of the file, which is also the class for all
            session generated from the file
    '''
    reader = Reader()
    data = reader.read(path)
    streams = list(data['tcp_stream'].drop_duplicates())
    sessions = []
    for stream in streams:
        packets = data.query('tcp_stream == @stream').reset_index(drop=True)
        sessions.append(Session(packets=packets,label=label))
    return sessions

class Vertex:
    def __init__(self, index, attr=None):
        '''
        Vertex object in state machine.
        Params:
            index: int,
            The index number of a vertex, ranged from 0 to STATE_NUM-1

            attr: list,
            The attribute vector attached to a vertex. Currently, it represents
            [First length, Average length]

            occur_time: int,
            The occurance of a vertex, used for attr-updating and other statistics.

            label: int,
            The label for vertex used in LSPG. 
        '''
        self.index = index
        self.attr = attr
        self.label = 0
        self.occur_time = 0

    def add(self, data):
        '''
        Add attribute vector, if self.attr is None, set self.attr=attr,
        otherwise, the Average length will be updated in the manner:
        '''
        # Extensibility, thanks to hashing technique, the label generation is
        # quite fast, up to O(n) where n is the num of elements within a vector.
        if self.attr is None:
            self.occur_time = 1
            self.attr = np.array([0, 0, 0, 0]).astype(np.float32)
            self.attr[0] = float(data[0])
            self.attr[1] = float(data[0])
            self.attr[2] = float(data[0])
            self.attr[3] = float(data[0])
        else: 
            self.occur_time += 1
            self.attr[1] = ((self.occur_time-1)*self.attr[1] + float(data[0]))/self.occur_time
            self.attr[2] = max(self.attr[2], float(data[0]))
            self.attr[3] = min(self.attr[3], float(data[0]))


class StateMachine:
    def __init__(self, vertices: list, adj: np.array, label: int):
        '''
        Params:
            vertices,
                The vertices whose attr is not None.
            adj: np.array,
                The weighted adjacency matrix.
            label: int,
                The class of the session generating SM.
        '''
        self.vertices = vertices
        self.adj = adj
        self.label = label

    def hashing(self, r:np.array):
        '''
        Map each of the attribute vector to a single discrete label.
        Args:
            r: np.array,
            Direction vector.
        '''
        for i in range(len(self.vertices)):
            self.vertices[i].label = angle_hash(self.vertices[i].attr, r)
            

    def showAdjacent(self):
        '''
        Visualize self.adj in heatmap.
        '''
        plt.figure(figsize=(8,6))
        plt.matshow(self.adj, fignum=1)

    def printVertices(self):
        '''
        Print the information of vertices that occur.
        '''
        for v in self.vertices:
            print("Vertex {}, attribute: {}, occur_time: {}, label: {}".format(
                v.index,
                v.attr,
                v.occur_time,
                v.label
            ))

    def printSM(self):
        '''
        Print basical information of the state machine.
        '''
        print("State Machine of class {}, {} vertices activated. Adjacency matrix of shape {}."
              .format(self.label, len(self.vertices), self.adj.shape))


def sessionToSM(session: Session, normalize=True, isolate=False, attr_normalize=True):
    '''
    Generate a state machine from a session.
    Args:
        session: Session
        normalize: boolean,
            Whether to scale the weights into interval [0,1]
        isolate: boolean:
            If True, keep all the isolated vertices(For test only)
        attr_normalize:
            If True, normalize all the attribute vectors of each vertex.
    '''
    adj = np.zeros(shape=(STATE_NUM, STATE_NUM))
    vertices = [Vertex(index=i, attr=None) for i in range(STATE_NUM)]
    before_index = 0
    after_index = 0
    pair = [0, 0] # (Empty, Empty)
    for idx, row in session.packets.iterrows():
        content_type = row['content_type']
        handshake_type = row['handshake_type']
        # data used to update the attr vector of vertex
        data = [row[key] for key in ATTR]
        # The TLS packet state.
        try:
            state = injection_tls((content_type, handshake_type))
        except:
            print("Exception occurs when handling session {}".format(session.identifier),
                  "At packet {}".format(idx))
            raise TypeError
        pair[0] = pair[1]
        pair[1] = state
        before_index = after_index
        after_index = cantor_index(pair[0], pair[1])
        vertices[after_index].add(data)
        if idx > 0:
            adj[before_index, after_index] += 1
    vertices = list(filter(lambda x: x.attr is not None, vertices))
    if normalize:
        adj /= session.len
    # We delete those isolated vertices
    if not isolate:
        non_zero_idx_row = set(np.unique(np.nonzero(adj)[0]))
        non_zero_idx_col = set(np.unique(np.nonzero(adj)[1]))
        zero_row = set(list(range(STATE_NUM)))-non_zero_idx_row
        zero_col = set(list(range(STATE_NUM)))-non_zero_idx_col
        zero_idx = zero_row & zero_col
        idx = list(set(list(range(STATE_NUM))) - zero_idx)
        adj = (adj[idx, :])[:, idx]
    for i in range(len(vertices)):
        vertices[i].index = i
    
    if attr_normalize:
        for i in range(len(vertices)):
            vertices[i].attr /= np.sum(vertices[i].attr)

    SM = StateMachine(
                vertices=vertices, 
                adj=adj, 
                label=session.label
                )
    return SM

def sessionsToSMs(sessions: list, normalize=True, isolate=False, attr_normalize=False):
    """
    A wrapper function that converts a group of sessions to SMs
    Args:
        sessions: list,
        A list of sessions.
    """
    SMs = []
    for s in sessions:
        SMs.append(sessionToSM(s, normalize, isolate, attr_normalize))

    return SMs


################################################################################################
#                                                                                              #
#                      First Order State Machine(For comparison only)                          #
#                                                                                              #
################################################################################################

FIRST_STATE_NUM = 17

def sessionToFirstOrderSM(session: Session, normalize=True, isolate=False, attr_normalize=True):
    '''
    Generate a state machine from a session.
    Args:
        session: Session
        normalize: boolean,
            Whether to scale the weights into interval [0,1]
        isolate: boolean:
            If True, keep all the isolated vertices(For test only)
        attr_normalize:
            If True, normalize all the attribute vectors of each vertex.
    '''
    adj = np.zeros(shape=(FIRST_STATE_NUM, FIRST_STATE_NUM))
    vertices = [Vertex(index=i, attr=None) for i in range(FIRST_STATE_NUM)]
    for idx, row in session.packets.iterrows():
        content_type = row['content_type']
        handshake_type = row['handshake_type']
        # data used to update the attr vector of vertex
        data = [row[key] for key in ATTR]
        before_index = after_index = 0
        # The TLS packet state.
        try:
            state = injection_tls((content_type, handshake_type))
        except:
            print("Exception occurs when handling session {}".format(session.identifier),
                  "At packet {}".format(idx))
            raise TypeError
        before_index = after_index
        after_index = state
        vertices[after_index].add(data)
        if idx > 0:
            adj[before_index, after_index] += 1
    vertices = list(filter(lambda x: x.attr is not None, vertices))
    if normalize:
        adj /= session.len
    # We delete those isolated vertices
    if not isolate:
        non_zero_idx_row = set(np.unique(np.nonzero(adj)[0]))
        non_zero_idx_col = set(np.unique(np.nonzero(adj)[1]))
        zero_row = set(list(range(FIRST_STATE_NUM)))-non_zero_idx_row
        zero_col = set(list(range(FIRST_STATE_NUM)))-non_zero_idx_col
        zero_idx = zero_row & zero_col
        idx = list(set(list(range(FIRST_STATE_NUM))) - zero_idx)
        adj = (adj[idx, :])[:, idx]
    for i in range(len(vertices)):
        vertices[i].index = i
    
    if attr_normalize:
        for i in range(len(vertices)):
            vertices[i].attr /= np.sum(vertices[i].attr)

    SM = StateMachine(
                vertices=vertices, 
                adj=adj, 
                label=session.label
                )
    return SM

def sessionToFirstOrderSMs(sessions: Session, normalize=True, isolate=False, attr_normalize=True):
    """
    A wrapper function that converts a group of sessions to First Order SMs
    Args:
        sessions: list,
        A list of sessions.
    """
    SMs = []
    for s in sessions:
        SMs.append(sessionToFirstOrderSM(s, normalize, isolate, attr_normalize))

    return SMs
