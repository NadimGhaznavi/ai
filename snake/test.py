from LinearQNet import Linear_QNet

my_model = Linear_QNet(40,
                       30, 1, 
                       40, 0,
                       50, 0,
                       3, 
                       100)
print(my_model)

my_model.insert_layer(2)
my_model.insert_layer(3)
print(my_model)
