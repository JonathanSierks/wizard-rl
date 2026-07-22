# tbc
obs_dim = 60 + 60 + 180 + 5 + 9 + (1 + 1 + 1)
max_bid = 20
hidden_dim = 256

net = WizardNet(obs_dim, max_bid, hidden_dim)             # EIN Netz
agent = RLAgent(net)                            # der Agent hält es
players = [Player(f"p{i}", i, agent) for i in range(3)]   # SELF-PLAY: derselbe Agent
game = Game(players)

for episode in range(100_000):
    game = Game(players)
    game.run()                          # spielt ein volles Spiel, Agent entscheidet
    batch = agent.drain_buffer()        # gesammelte (obs, action, mask, return)
    # 
    loss = update(net, optimizer, batch)  # EIN Gradientenschritt über das Netz

