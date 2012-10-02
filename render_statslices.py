#!/usr/bin/python

#-------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#-------------------------------------------------------------------------------

from IPython.core.debugger import Tracer; debug_here = Tracer()

import os, argparse, sys
import numpy as np

import aizkolari_utils as au

#${aizkod}/render_statslices.py --bg $bg --s1 $sg -o test --s1_min 0.01

def set_parser():
   parser = argparse.ArgumentParser(description='Saves a gif or png image file containing the slices from FSL Renderstats (overlay), but adapted to my needs. The slices file has been modified: check ./slices. You also need imagemagick installed.')
   parser.add_argument('-b', '--bg', dest='bg_img', required=True,
                      help='background image file')
   parser.add_argument('--s1', dest='stat1_img', required=True,
                      help='stat image 1')
   parser.add_argument('--s1_min', dest='stat1_min', required=False, type=float,
                      help='Minimum intensity value used to range the plotted stats. Will use volume minimum if not set.')
   parser.add_argument('--s1_max', dest='stat1_max', required=False, type=float,
                      help='Maximum intensity value used to range the plotted stats. Will use volume maximum if not set.')
   parser.add_argument('-o', '--out', dest='out_img', required=False,
                      help='Output image file.', default='out')
   parser.add_argument('--nobar', dest='nobar', required=False, action='store_true',
                      help='If set, no bar will be appended to the image.', default=False)

   return parser
#-------------------------------------------------------------------------------

def get_image_size(fname):
   return au.exec_comm(['convert', fname, '-format', '%G', '-identify', 'null:']).strip().split('x')


#-------------------------------------------------------------------------------
## START EXTRACT FEATSET
#-------------------------------------------------------------------------------
def main(argv=None):

   #parsing args
   parser = set_parser()

   try:
      args = parser.parse_args ()
   except argparse.ArgumentError, exc:
      print (exc.message + '\n' + exc.argument)
      parser.error(str(msg))
      return 0

   bg_img = args.bg_img.strip    ()
   s1_img = args.stat1_img.strip ()
   ot_img = args.out_img.strip   ()

   s1_min = args.stat1_min
   s1_max = args.stat1_max
   nobar  = args.nobar

   #setting slices
   slices = 'slices'
   if os.path.exists('/usr/local/bin/slices'):
      slices = '/usr/local/bin/slices'

   #reading input
   bg_range = au.get_volume_intrange(bg_img)
   bg_min   = bg_range[0]
   bg_max   = bg_range[1]

   s1_range = au.get_volume_intrange(s1_img)

   if not s1_min:
      s1_min = s1_range[0]

   if not s1_max:
      s1_max = s1_range[1]

   #get a temporal place to write stuff
   statsvol = au.exec_comm('tmpnam').strip()
   statsbar = au.exec_comm('tmpnam').strip()
   #if not os.path.exists(tmpd):
   #   exec_comm(['mkdir', tmpd])

   #statsvol = tmpd + os.path.sep + 'outvol'
   #statsbar = tmpd + os.path.sep + 'outbar'

   #overlay command
   overlay = 'overlay 1 0 ' + bg_img + ' ' + str(bg_min) + ' ' + str(bg_max) + ' ' + s1_img + ' ' + str(s1_min) + ' ' + str(s1_max) + ' ' + statsbar + ' y ' + statsvol
   au.exec_comm (overlay.split())

   #slices command
   olddir = os.getcwd()

   outdir = os.path.dirname(os.path.abspath(ot_img))

   os.chdir(outdir)

#   debug_here()

   slicef = au.exec_comm([slices, statsvol, '1']).strip()

   #debug_here()

   if not (nobar):
      #resize and append bar
      statsvol_size = get_image_size(slicef)
      nusize        = np.floor(int(statsvol_size[0])/2)
      au.exec_comm(['mogrify', '-resize', 'x' + str(nusize), statsbar])
      au.exec_comm(['pngappend', slicef, '+', statsbar, ot_img])
   else:
      au.exec_comm(['cp', slicef, ot_img])

   #manage outputs
   au.exec_comm(['rm', slicef])
   au.exec_comm(['rm', statsbar])

   os.chdir(olddir)

   print ('Stats image created in ' + ot_img)
#-------------------------------------------------------------------------------
## END RENDER STAT SLICES
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())


