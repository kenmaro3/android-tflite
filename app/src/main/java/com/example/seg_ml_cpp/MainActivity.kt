package com.example.seg_ml_cpp

import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.*
import androidx.appcompat.app.AppCompatActivity
import android.util.Log
import com.example.seg_ml_cpp.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.android.Utils.bitmapToMat
import org.opencv.android.Utils.matToBitmap
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

import android.view.PixelCopy
import android.view.SurfaceView
import android.widget.*
import androidx.annotation.RequiresApi

import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.w3c.dom.Text


class MainActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var imageView2: ImageView
    private lateinit var videoView: VideoView
    private lateinit var button: Button
    private lateinit var textView: TextView
    private lateinit var textView2: TextView
    private lateinit var textView3: TextView

    private lateinit var interpreter: Interpreter

    private var predictTimeStart = 0L
    private var predictTimeStop = 0L
    private var timeDiff = 0L

    private lateinit var binding: ActivityMainBinding

    private fun getBitmapFromAsset(strName: String): Bitmap? {
        val assetManager = assets
        var istr: InputStream? = null
        try {
            istr = assetManager.open(strName)
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return BitmapFactory.decodeStream(istr)
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        var fileDescriptor: AssetFileDescriptor = this.assets.openFd(modelName)
        var inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        var fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    @RequiresApi(Build.VERSION_CODES.N)
    fun usePixelCopy(videoView: SurfaceView, callback: (Bitmap?) -> Unit) {
        val bitmap: Bitmap = Bitmap.createBitmap(
            videoView.width,
            videoView.height,
            Bitmap.Config.ARGB_8888
        );
        try {
            // Create a handler thread to offload the processing of the image.
            val handlerThread = HandlerThread("PixelCopier");
            handlerThread.start();
            PixelCopy.request(
                videoView, bitmap,
                PixelCopy.OnPixelCopyFinishedListener { copyResult ->
                    if (copyResult == PixelCopy.SUCCESS) {
                        callback(bitmap)
                    }
                    handlerThread.quitSafely();
                },
                Handler(handlerThread.looper)
            )
        } catch (e: IllegalArgumentException) {
            callback(null)
            // PixelCopy may throw IllegalArgumentException, make sure to handle it
            e.printStackTrace()
        }
    }

    private fun predict_from_bitmap(bitmapInput: Bitmap){


        var mat1 = Mat()
        var mat2 = Mat()
//        var mat3 = Mat()
        val ppTimeStart = SystemClock.uptimeMillis()
        bitmapToMat(bitmapInput, mat1)
//        processImage(mat1.nativeObjAddr, mat3.nativeObjAddr)
        Imgproc.resize(mat1, mat2, Size(128.0, 128.0))
        var bitmap = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        matToBitmap(mat2, bitmap)

        val input = ByteBuffer.allocateDirect(128*128*3*4).order(ByteOrder.nativeOrder())
        for (y in 0 until 128) {
            for (x in 0 until 128) {
                val px = bitmap.getPixel(x, y)

                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF

                val rf = r / 255f
                val gf = g / 255f
                val bf = b / 255f

                input.putFloat(rf)
                input.putFloat(gf)
                input.putFloat(bf)
            }
        }

        val bufferSize = 128 * 128 * java.lang.Float.SIZE / java.lang.Byte.SIZE
        val modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder())
        val ppTimeStop = SystemClock.uptimeMillis()
        val ppTimeDiff = ppTimeStop - ppTimeStart
        Handler(mainLooper).post({
            textView2.setText("PP Time: $ppTimeDiff")
        })
        // Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors.
        interpreter?.run(input, modelOutput)
        val popTimeStart = SystemClock.uptimeMillis()

        var floatArrayOutput = FloatArray(128*128)

        for (i in 0..127){
            for (j in 0..127){
                floatArrayOutput[i*128 + j] = modelOutput.getFloat((i*128+j)*4)
            }
        }

        val byteBufferParsedResult = ByteBuffer.allocateDirect(128*128*4).order(ByteOrder.nativeOrder())
        byteBufferParsedResult.rewind()

        for(i in 0..127){
            for(j in 0..127){
                byteBufferParsedResult.put(0.toByte()) // B
                byteBufferParsedResult.put(0.toByte()) // G
                if(floatArrayOutput[i*128+j] > 0.3){
                    byteBufferParsedResult.put(255.toByte()) // R
                }
                else{
                    byteBufferParsedResult.put(0.toByte()) // Red
                }
                byteBufferParsedResult.put(255.toByte()) // A
            }
        }
        byteBufferParsedResult.rewind()

        val parsedBitmapResult = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        parsedBitmapResult.copyPixelsFromBuffer(byteBufferParsedResult)
        var parsedBitmapResultResized = Bitmap.createScaledBitmap(parsedBitmapResult, 640, 480, true)
        var bitmapInputResized = Bitmap.createScaledBitmap(bitmapInput, 640, 480, true)
        val popTimeStop = SystemClock.uptimeMillis()
        val popTimeDiff = popTimeStop - popTimeStart
        Handler(mainLooper).post({
            textView3.setText("POP Time: $popTimeDiff")
        })


        Handler(mainLooper).post({
            imageView.setImageBitmap(bitmapInputResized)
            imageView2.setImageBitmap(parsedBitmapResultResized)
        })

    }

    private fun findView(){
        imageView = findViewById(R.id.imageView)
        imageView2 = findViewById(R.id.imageView2)
        button = findViewById(R.id.button)
        videoView = findViewById(R.id.videoView)
        textView = findViewById(R.id.textView)
        textView2 = findViewById(R.id.textView2)
        textView3 = findViewById(R.id.textView3)
    }

    @RequiresApi(Build.VERSION_CODES.N)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

//        val delegate = GpuDelegate(GpuDelegate.Options().setQuantizedModelsAllowed(true)) // DEQUANTIZE not supported
        val delegate = GpuDelegate(GpuDelegate.Options().setQuantizedModelsAllowed(false)) // TRANSPOSE_CONV: Max version supported: 2. Requested version 3.
        val options = Interpreter.Options().addDelegate(delegate).setNumThreads(4)
//        val options = Interpreter.Options().setUseNNAPI(true) // 1200
//        val options = Interpreter.Options().setNumThreads(4) //500
//        val options = Interpreter.Options().setUseNNAPI(true).setNumThreads(4) // 500

//        val options = Interpreter.Options()

        val tmpFile = loadModelFile("model_128_16_1_metadata.tflite") //okay
//        val tmpFile = loadModelFile("model_128_dr_1_metadata.tflite") // NOT Okay: Internal error: Cannot create interpreter: Didn't find op for builtin opcode 'CONV_2D' version '5'
        interpreter = Interpreter(tmpFile, options)

        findView()

        OpenCVLoader.initDebug()

        button.setOnClickListener{
            usePixelCopy(videoView) { bitmap: Bitmap? ->
                if(bitmap != null){
                    predictTimeStart = SystemClock.uptimeMillis()
                    predict_from_bitmap(bitmap)
                    predictTimeStop = SystemClock.uptimeMillis()
                    timeDiff = predictTimeStop - predictTimeStart
                    Handler(mainLooper).post({
                        textView.setText("Time: $timeDiff")
                    })
                }
            }
        }


        Handler(mainLooper).postDelayed({
            var moviePath = Uri.parse("android.resource://" + packageName + "/" + R.raw.eval0)
            videoView.setVideoURI(moviePath)

            videoView.setOnPreparedListener{
                videoView.start()

                videoView.setMediaController(MediaController(this))
            }

            videoView.setOnCompletionListener {

            }
        }, 200)


        // Example of a call to a native method
        binding.sampleText.text = stringFromJNI()

        var tmpBitmap = getBitmapFromAsset("eval_0050.jpg")
        predict_from_bitmap(tmpBitmap!!)


        Log.d("mydebug", "test")

    }

    /**
     * A native method that is implemented by the 'seg_ml_cpp' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String
    external fun processImage(objMatSrc: Long, objMatDst: Long): Int

    companion object {
        // Used to load the 'seg_ml_cpp' library on application startup.
        init {
            System.loadLibrary("seg_ml_cpp")
        }
    }


}