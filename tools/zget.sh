#!/bin/sh
# zget.sh NAME: fetch NAME.zip (one dataset's spectrogram bundle, e.g. candombe) from
# Zenodo record 13922116 (Beat This! spectrograms, CC BY 4.0) in 8 parallel byte ranges,
# check its md5, unpack the .npz into data/beatthis_spect and link it where the
# beat_this data module looks: data/bt/audio/spectrograms/NAME.npz.
# NAME=beat_this_annotations fetches the annotations; move them to data/bt/annotations.
set -e
cd "$(dirname "$0")/.."
DATA="${TSM_DATA:-$(pwd)/data}"
PY="${PY:-python}"
D="$DATA/beatthis_spect"
mkdir -p "$D" "$DATA/bt/audio/spectrograms"
cd "$D"
U="https://zenodo.org/records/13922116/files/$1.zip?download=1"
S=$(curl -sIL "$U" | grep -i content-length | tail -1 | tr -dc 0-9)
# 8 ranges at once: Zenodo throttles single connections to a few hundred KB/s
N=8; C=$(( (S + N - 1) / N ))
for i in 0 1 2 3 4 5 6 7; do a=$((i*C)); b=$(( (i+1)*C - 1 )); [ $b -ge $S ] && b=$((S-1)); curl -sL --retry 5 -r $a-$b -o $1.p$i "$U" & done
wait
cat $1.p0 $1.p1 $1.p2 $1.p3 $1.p4 $1.p5 $1.p6 $1.p7 > $1.zip && rm $1.p?
WANT=$(curl -s https://zenodo.org/api/records/13922116 | $PY -I -c "import json,sys; print([f['checksum'][4:] for f in json.load(sys.stdin)['files'] if f['key']=='$1.zip'][0])")
GOT=$(md5sum $1.zip | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { echo "$1 md5 mismatch"; exit 1; }
$PY -I -c "import zipfile; zipfile.ZipFile('$1.zip').extractall('.')"
rm $1.zip
if [ -f "$1.npz" ]; then ln -sf "$D/$1.npz" "$DATA/bt/audio/spectrograms/$1.npz"; fi
echo "$1 ok"
