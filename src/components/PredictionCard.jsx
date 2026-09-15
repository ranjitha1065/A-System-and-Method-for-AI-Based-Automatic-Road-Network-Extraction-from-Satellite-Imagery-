import { lazy, Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import JSZip from 'jszip';
import toast from 'react-hot-toast';
import { Download, FileArchive, LoaderCircle, Sparkles } from 'lucide-react';
import UploadCard from './UploadCard';
import LoadingExperience from './LoadingExperience';
import StatsPanel from './StatsPanel';
import HistoryPanel from './HistoryPanel';
import PDFReportButton from './PDFReportButton';

const ComparisonView = lazy(() => import('./ComparisonView'));

const API_URL = `${import.meta.env.VITE_API_URL}/predict`;
const HISTORY_KEY = 'roadai-last-predictions';

/* -------------------------------------------------------
   Download helper
------------------------------------------------------- */
const download = (url, name) => {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();

  toast.success('Download complete');
};

/* -------------------------------------------------------
   Convert image to a smaller JPEG before upload.

   This prevents Vercel 413 Payload Too Large errors.
------------------------------------------------------- */
const prepareImageForUpload = async (file) => {
  const MAX_DIMENSION = 1024;
  const MAX_FILE_SIZE = 3.5 * 1024 * 1024; // 3.5 MB
  const JPEG_QUALITY = 0.80;

  // If already a reasonably small JPEG/PNG, use original.
  if (
    file.size <= MAX_FILE_SIZE &&
    (file.type === 'image/jpeg' || file.type === 'image/png')
  ) {
    return file;
  }

  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);

    image.onload = () => {
      try {
        let width = image.naturalWidth;
        let height = image.naturalHeight;

        // Keep aspect ratio while limiting maximum dimension.
        if (Math.max(width, height) > MAX_DIMENSION) {
          const scale = MAX_DIMENSION / Math.max(width, height);
          width = Math.round(width * scale);
          height = Math.round(height * scale);
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const context = canvas.getContext('2d');

        if (!context) {
          URL.revokeObjectURL(objectUrl);
          reject(new Error('Unable to create image canvas.'));
          return;
        }

        context.drawImage(image, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            URL.revokeObjectURL(objectUrl);

            if (!blob) {
              reject(new Error('Unable to compress image.'));
              return;
            }

            // If compression somehow still produces a large file,
            // reduce quality once more.
            if (blob.size > MAX_FILE_SIZE) {
              canvas.toBlob(
                (smallerBlob) => {
                  if (!smallerBlob) {
                    reject(new Error('Unable to reduce image size.'));
                    return;
                  }

                  resolve(
                    new File(
                      [smallerBlob],
                      file.name.replace(/\.[^/.]+$/, '') + '.jpg',
                      {
                        type: 'image/jpeg',
                        lastModified: Date.now(),
                      }
                    )
                  );
                },
                'image/jpeg',
                0.65
              );
            } else {
              resolve(
                new File(
                  [blob],
                  file.name.replace(/\.[^/.]+$/, '') + '.jpg',
                  {
                    type: 'image/jpeg',
                    lastModified: Date.now(),
                  }
                )
              );
            }
          },
          'image/jpeg',
          JPEG_QUALITY
        );
      } catch (error) {
        URL.revokeObjectURL(objectUrl);
        reject(error);
      }
    };

    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(
        new Error(
          'This image format cannot be processed by the browser.'
        )
      );
    };

    image.src = objectUrl;
  });
};

/* -------------------------------------------------------
   Calculate pixels from predicted mask
------------------------------------------------------- */
const pixelsFromMask = async (mask) =>
  new Promise((resolve, reject) => {
    const image = new Image();

    image.onload = () => {
      const canvas = document.createElement('canvas');

      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;

      const context = canvas.getContext('2d');

      if (!context) {
        reject(new Error('Unable to read prediction mask.'));
        return;
      }

      context.drawImage(image, 0, 0);

      const values = context.getImageData(
        0,
        0,
        canvas.width,
        canvas.height
      ).data;

      let road = 0;

      for (let index = 0; index < values.length; index += 4) {
        if (values[index] > 127) {
          road++;
        }
      }

      resolve({
        road,
        total: values.length / 4,
        width: image.naturalWidth,
        height: image.naturalHeight,
      });
    };

    image.onerror = () => {
      reject(new Error('Unable to read prediction mask.'));
    };

    image.src = mask;
  });

/* -------------------------------------------------------
   Convert Base64/data URL to Blob
------------------------------------------------------- */
const base64Blob = (url) =>
  fetch(url).then((response) => response.blob());

/* -------------------------------------------------------
   Main component
------------------------------------------------------- */
export default function PredictionCard() {
  const [file, setFile] = useState();
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState(0);
  const [result, setResult] = useState();
  const [error, setError] = useState('');
  const [stats, setStats] = useState();

  const [history, setHistory] = useState(() =>
    JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  );

  const preview = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file]
  );

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }, [history]);

  /* -------------------------------------------------------
     Select uploaded file
  ------------------------------------------------------- */
  const selectFile = useCallback((selected) => {
    setFile(selected);
    setResult(undefined);
    setStats(undefined);
    setError('');

    toast.success('Upload successful');
  }, []);

  /* -------------------------------------------------------
     Build prediction statistics
  ------------------------------------------------------- */
  const buildStats = useCallback(async (data, name, elapsed) => {
    const detail = await pixelsFromMask(data.mask);

    const coverage = (detail.road / detail.total) * 100;

    const roadLength = detail.road * 0.5;

    return {
      roadPixels: detail.road.toLocaleString(),

      backgroundPixels: (
        detail.total - detail.road
      ).toLocaleString(),

      coverage: `${coverage.toFixed(2)}%`,

      roadLength:
        roadLength > 1000
          ? `${(roadLength / 1000).toFixed(2)} km`
          : `${roadLength.toFixed(1)} m`,

      size: `${detail.width} × ${detail.height}`,

      time: `${(elapsed / 1000).toFixed(2)} s`,

      model: 'U-Net Fine Tuned',

      confidence: `${Math.round(
        (data.confidence || 0) * 100
      )}%`,

      metadata: {
        filename: name,
        timestamp: new Date().toISOString(),
        roadCoverage: coverage,
        roadPixels: detail.road,
        estimatedRoadLengthMeters: roadLength,
        imageWidth: detail.width,
        imageHeight: detail.height,
        model: 'U-Net Fine Tuned',
        predictionTime: elapsed,
      },
    };
  }, []);

  /* -------------------------------------------------------
     Run AI prediction
  ------------------------------------------------------- */
  const predict = useCallback(async () => {
    if (!file) return;

    setLoading(true);
    setStage(0);
    setError('');
    setResult(undefined);

    const timer = performance.now();

    const progression = setInterval(() => {
      setStage((value) => Math.min(2, value + 1));
    }, 900);

    try {
      /*
       * IMPORTANT:
       * Compress/resize image before sending to Vercel.
       */
      const uploadFile = await prepareImageForUpload(file);

      console.log(
        `Original image size: ${(file.size / 1024 / 1024).toFixed(2)} MB`
      );

      console.log(
        `Upload image size: ${(uploadFile.size / 1024 / 1024).toFixed(2)} MB`
      );

      const form = new FormData();

      form.append('image', uploadFile);

      const { data } = await axios.post(
        API_URL,
        form
      );

      if (!data.success) {
        throw new Error(data.error || 'Prediction failed.');
      }

      const computed = await buildStats(
        data,
        file.name,
        performance.now() - timer
      );

      setResult(data);
      setStats(computed);

      setHistory((items) =>
        [
          {
            ...data,
            id: crypto.randomUUID(),
            filename: file.name,
            createdAt: Date.now(),
            stats: computed,
          },
          ...items,
        ].slice(0, 5)
      );

      toast.success('Prediction completed');
    } catch (requestError) {
      console.error('Prediction error:', requestError);

      if (requestError.response?.status === 413) {
        setError(
          'Image is too large. Please upload a smaller satellite image.'
        );

        toast.error('Image is too large');
      } else if (requestError.response?.data?.error) {
        setError(requestError.response.data.error);
        toast.error('Prediction failed');
      } else if (requestError.request) {
        setError('Unable to connect to AI server.');
        toast.error('AI server connection failed');
      } else {
        setError(
          requestError.message || 'Prediction failed.'
        );

        toast.error('Prediction failed');
      }
    } finally {
      clearInterval(progression);
      setLoading(false);
    }
  }, [buildStats, file]);

  /* -------------------------------------------------------
     Export ZIP
  ------------------------------------------------------- */
  const exportZip = useCallback(async () => {
    if (!result || !stats) return;

    const zip = new JSZip();

    zip.file(
      'original.png',
      await base64Blob(result.original)
    );

    zip.file(
      'mask.png',
      await base64Blob(result.mask)
    );

    zip.file(
      'overlay.png',
      await base64Blob(result.overlay)
    );

    zip.file(
      'prediction.json',
      JSON.stringify(
        stats.metadata,
        null,
        2
      )
    );

    const blob = await zip.generateAsync({
      type: 'blob',
    });

    download(
      URL.createObjectURL(blob),
      'roadai-results.zip'
    );
  }, [result, stats]);

  /* -------------------------------------------------------
     Keyboard shortcuts
  ------------------------------------------------------- */
  useEffect(() => {
    const key = (event) => {
      if (
        event.ctrlKey &&
        event.key === 'Enter'
      ) {
        event.preventDefault();
        predict();
      }

      if (
        event.ctrlKey &&
        event.key.toLowerCase() === 'd' &&
        result
      ) {
        event.preventDefault();
        download(
          result.overlay,
          'overlay.png'
        );
      }
    };

    window.addEventListener(
      'keydown',
      key
    );

    return () =>
      window.removeEventListener(
        'keydown',
        key
      );
  }, [predict, result]);

  /* -------------------------------------------------------
     Restore prediction
  ------------------------------------------------------- */
  const restore = (item) => {
    setResult(item);
    setStats(item.stats);
    setFile(undefined);
    setError('');

    toast.success('Prediction restored');
  };

  return (
    <section className="glass rounded-[2rem] p-5 sm:p-8">

      <div className="mb-7 flex flex-wrap items-center justify-between gap-3">

        <div>
          <p className="text-xs font-bold uppercase tracking-[.2em] text-violet-300">
            Inference Studio
          </p>

          <h2 className="mt-2 text-2xl font-bold">
            Generate a road mask
          </h2>
        </div>

        <div className="rounded-xl bg-white/5 px-3 py-2 text-xs text-slate-400">
          U-Net Fine Tuned · Ready
        </div>

      </div>

      <UploadCard
        file={file}
        onFile={selectFile}
      />

      <button
        disabled={!file || loading}
        onClick={predict}
        className="gradient-button mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl px-5 py-3.5 text-sm font-bold disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? (
          <>
            <LoaderCircle
              className="animate-spin"
              size={18}
            />
            Analyzing image…
          </>
        ) : (
          <>
            <Sparkles size={17} />
            Run AI Prediction
          </>
        )}
      </button>

      {loading && (
        <LoadingExperience stage={stage} />
      )}

      {error && (
        <p className="mt-4 text-sm text-orange-200">
          {error}
        </p>
      )}

      {result && (
        <>
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">

            <button
              onClick={() =>
                download(
                  result.original,
                  'original.png'
                )
              }
              className="action-button"
            >
              <Download size={16} />
              Original
            </button>

            <button
              onClick={() =>
                download(
                  result.mask,
                  'predicted_mask.png'
                )
              }
              className="action-button"
            >
              <Download size={16} />
              Mask
            </button>

            <button
              onClick={() =>
                download(
                  result.overlay,
                  'overlay.png'
                )
              }
              className="action-button"
            >
              <Download size={16} />
              Overlay
            </button>

            {stats && (
              <PDFReportButton
                result={result}
                stats={stats}
              />
            )}

            <button
              onClick={exportZip}
              className="action-button"
            >
              <FileArchive size={16} />
              Export ZIP
            </button>

          </div>

          {stats && (
            <StatsPanel stats={stats} />
          )}

          <Suspense
            fallback={
              <LoadingExperience stage={2} />
            }
          >
            <ComparisonView
              result={result}
              preview={preview}
            />
          </Suspense>
        </>
      )}

      <HistoryPanel
        items={history}
        onRestore={restore}
        onClear={() => {
          setHistory([]);
          toast.success('History cleared');
        }}
      />

    </section>
  );
}