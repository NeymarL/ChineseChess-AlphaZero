# -*- mode: python -*-

block_cipher = None

a = Analysis(['play_games.py'],
             pathex=['C:\\Users\\niuhe\\Desktop\\ChineseChess-AlphaZero-master'],
             binaries=[],
             datas=[
                ('C:\\Users\\niuhe\\Desktop\\ChineseChess-AlphaZero-master\\cchess_alphazero\\play_games\\images\\WOOD.GIF', 'cchess_alphazero\\play_games\\images'),
                ('C:\\Users\\niuhe\\Desktop\\ChineseChess-AlphaZero-master\\cchess_alphazero\\play_games\\PingFang.ttc', 'cchess_alphazero\\play_games'),
                ('C:\\Users\\niuhe\\Desktop\\ChineseChess-AlphaZero-master\\cchess_alphazero\\play_games\\images\\WOOD\\*.GIF', 'cchess_alphazero\\play_games\\images\\WOOD')
              ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='play_games',
          debug=False,
          strip=False,
          upx=True,
          console=True , resources=['cchess_alphazero\\\\play_games\\\\images'])
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='play_games')
