class Edge:
    def __init__(self, u, v, attrs):
        self.set_properties(u, v, attrs)
        
    def set_properties(self, u, v, attrs):
        self.node_in = u
        self.node_out = v
        self.attrs = attrs
        self._set_type()
        
    def _set_type(self):
        # Devign format
        if "label" in self.attrs:
            self.type = self.attrs["label"]

        # Joern 2.x export format
        elif "type" in self.attrs:
            self.type = self.attrs["type"]

        else:
            self.type = "UNKNOWN"