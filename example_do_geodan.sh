#calculating jacobian geodesic anisotropy
d=geodan
aizkod=~/Dropbox/Documents/phd/work/aizkolari
id=/data/oasis_jesper_features/jacmat
od=/data/oasis_jesper_features/${d}
mask=/data/oasis_jesper_features/MNI152_T1_1mm_brain_mask_dil.nii.gz
cd $id
lst=`ls control*.nii.gz`
for i in $lst; do
   echo $i
   ifile=${id}/$i
   ofile=`remove_ext $i`
   ofile=${od}/${ofile}_${d}.nii.gz
   if [ ! -f $ofile ]; then
      ${aizkod}/matrans.py --in=${ifile} --out=${ofile} --mask=$mask --${d}
   fi
done
