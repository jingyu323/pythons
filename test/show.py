from comon import glovar
class Show:
  def showchange(self):
    print(glovar.x)
    if glovar.x!=1:
      print('show x change')
    else:
      print('show x=1')