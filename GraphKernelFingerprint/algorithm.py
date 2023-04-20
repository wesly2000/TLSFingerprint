import numpy as np

'''State mapping:
    (0, 0)      = Empty                                     0
    (20, None)  = Change Cipher Spec                        1
    (21, None)  = Alert                                     2
    (22, None)  = Handshake, Encrypted                      3
    (22, 0)     = Handshake, Hello Request                  4
    (22, 1)     = Handshake, Client Hello                   5
    (22, 2)     = Handshake, Server Hello                   6
    (22, 4)     = Handshake, NewSessionTicket               7
    (22, 11)    = Handshake, Certificate                    8
    (22, 12)    = Handshake, Server Key Exchange            9
    (22, 13)    = Handshake, Certificate Request            10
    (22, 14)    = Handshake, Server Hello Done              11
    (22, 15)    = Handshake, Certificate Verify             12
    (22, 16)    = Handshake, Client Key Exchange            13
    (22, 20)    = Handshake, Finished                       14
    (22, 22)    = Handshake, Certificate Status             15
    (23, None)  = Application Data                          16
    
    '''

cantor_pair = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
               40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
               59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 
               78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 
               97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
               113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
               129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
               145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 
               162, 163, 164, 165, 166, 167, 168, 169, 173, 174, 175, 176, 177, 178, 179, 180, 
               181, 182, 183, 184, 185, 186, 187, 193, 194, 195, 196, 197, 198, 199, 200, 201, 
               202, 203, 204, 205, 206, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 
               225, 226, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 259, 260, 
               261, 262, 263, 264, 265, 266, 267, 268, 269, 283, 284, 285, 286, 287, 288, 289, 
               290, 291, 292, 308, 309, 310, 311, 312, 313, 314, 315, 316, 334, 335, 336, 337, 
               338, 339, 340, 341, 361, 362, 363, 364, 365, 366, 367, 389, 390, 391, 392, 393, 
               394, 418, 419, 420, 421, 422, 448, 449, 450, 451, 479, 480, 481, 511, 512, 544]

state_map = {
    (0,0):      0,
    (20,None):  1,
    (21,None):  2,
    (22,None):  3,
    (22,0):     4,
    (22,1):     5,
    (22,2):     6,
    (22,4):     7,
    (22,11):    8,
    (22,12):    9,
    (22,13):    10,
    (22,14):    11,
    (22,15):    12,
    (22,16):    13,
    (22,20):    14,
    (22,22):    15,
    (23,None):  16,
}

def injection_tls(state:tuple):
    """ map the state (content_type, handshake_type) to a unique number.
    Parameter
    ---------
    content_type: str or int
    handshake_type: str or int

    Returns
    -------
    state_num: int, the unique number represents the tls state
    """
    content_type, handshake_type = state
    content_type = int(content_type)
    if handshake_type is not None:
        handshake_type = int(handshake_type)
    return state_map[(content_type, handshake_type)]

def cantor_index(k_1: int, k_2: int):
    '''
    Mapping N*N->N, with Cantor pairing rule.
    Args:
        k_1,k_2: int,
        Numbers to be mapped with.

    Return:
        k: int
        The order of pair (k_1,k_2)
    '''
    k = (k_1+k_2)*(k_1+k_2+1)//2 + k_2
    index = cantor_pair.index(k)
    return index

def angle_hash(v: np.array, r: np.array):
    '''
    Implement of LSH function under similarity measure sim(v,r)=1-ang(v,r)/pi,
            /--- 0, if v.r < 0
    h(v,r)=|
            \--- 1, o.w.
    Args:
        v: np.array,
        The attribute function that will be mapped to a single label.

        r: np.array,
        The direction vector draw from n dimension Gaussian.
    '''
    return 'a' if np.dot(v,r) >= 0 else 'b'

class LSH:
    def __init__(self, hash_function=angle_hash) -> None:
        self.hash = hash_function

    def hashing(self, *args):
        return self.hash(*args)
    
def dirac_kernel(a, b):
    '''
    The similarity metric, usually used for discrete value.
    '''
    return a == b

def brownian_kernel(a: float, b: float, c=3.0):
    '''
    The similarity metric, usually used for continuous value.
    '''
    return max(0, c-abs(a-b))

def weighted_f1(recall:list, precision:list, beta=2):
    assert len(recall) == len(precision), "The length of recall and precision must be equal."
    weight_f1 = [round((1+beta**2)*p*r/(beta**2*p+r),4) for r,p in zip(recall, precision)]
    return weight_f1
