class AblationOption:
  def __init__(self, normalization_layer_removal, max_pool, hidden_layer_removal):
    self.normalization_layer_removal = normalization_layer_removal
    self.max_pool = max_pool
    self.hidden_layer_removal = hidden_layer_removal

#  def __init__(self, normalization_layer_removal, max_pool):
#    self.normalization_layer_removal = normalization_layer_removal
#    self.max_pool = max_pool
