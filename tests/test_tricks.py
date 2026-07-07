import pytest
from cards import Card
from tricks import resolve_winner

@pytest.mark.parametrize("plays, trump, gewinner", [
    ([(0,Card("red",5)),  (1,Card("red",9))],  "green", 1), # same color, higher card wins
    ([(0,Card("red",5)),  (1,Card("blue",9))],  "green", 0), # table color (first color to be played) wins over other colors
    ([(0,Card("red",5)),  (1,Card("green",2))], "green", 1), # trump wins
    ([(0,Card("blue",13)),(1,Card("green",14)),(2,Card("red",0))],"green", 1), # wizard wins
    ([(0,Card("blue",14)),(1,Card("green",14))],"green", 0), # first wizard wins
    ([(0,Card("blue",0)),(1,Card("green",0)), (2,Card("red",0))],"green", 0), # multiple joker, first joker wins
    # tbd: was wenn als trump wizard/narr bzw. keiner in der letzten runde?
    ([(0,Card("red",0)),(1,Card("blue",8)), (2,Card("yellow",9))],"green", 1), # wenn als erstes Wizard --> nächste karte bestimmt lead_color
    # tbd: wenn als erstes Narr, legt das nicht die table color fest. die nächste normale karte macht das
    
], ids=["same color, higher card wins", "table color wins", "trump wins", "wizard wins", "first wizard wins", "first joker wins", "erste karte narr"])

def test_stich_gewinner(plays, trump, gewinner):
    assert resolve_winner(plays, trump) == gewinner