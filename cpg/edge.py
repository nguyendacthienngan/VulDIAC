class Edge:
    def __init__(self, u, v, attrs):
        self.set_properties(u, v, attrs)
        
    def set_properties(self, u, v, attrs):
        self.node_in = u
        self.node_out = v
        self.attrs = attrs
        self._set_type()
        
    def _set_type(self):
        self.type = self.attrs["label"]
        
        