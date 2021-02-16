
def config_layers2(self):
     ''' decide '''
      n_prev = self.layers[0].n_nodes
       for i in range(1, len(self.layers)):
            self.layers[i].n_prev = n_prev
            n_prev = self.layers[i].n_nodes
        self.layers.reverse()
        n_next = self.layers[0].n_nodes
        for i in range(1, len(self.layers)):
            self.layers[i].n_next = n_next
            n_next = self.layers[i].n_nodes
        self.layers.reverse()
        for layer in self.layers:
            layer.weights = np.random.uniform(0, 1.0, (layer.n_prev, layer.n_next))
        for layer in self.layers:
            print(layer.n_prev, layer.n_prev)

# May be changed, else clause (?)
            # layer.n_next = self.layers[i+1].n_nodes if (i+1) in index_range else layer.n_nodes
