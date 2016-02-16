package haiqing.word2vec

import java.io._
import scala.compat.Platform._
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuilder
import scala.io.Source
import com.github.fommil.netlib.BLAS.{getInstance => blas}
/**
 * Created by hwang on 09.02.16.
 */
class SenseAssignment extends Serializable {

  private var vectorSize = 100
  private var minCount = 5
  private var iterations = 2
  private var window = 5
  private var negative = 5
  private var learningRate = 0.025f
  private var seed = 42l
  private var numRDDs = 5
  private var maxAdjusting = 10
  private var local = false


  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }
  def setIterations(iterations: Int): this.type = {
    this.iterations = iterations
    this
  }
  def setWindow(window: Int): this.type = {
    this.window = window
    this
  }
  def setNegative(negative: Int): this.type = {
    this.negative = negative
    this
  }
  def setLearningRate(learningRate: Int): this.type = {
    this.learningRate = learningRate
    this
  }
  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }
  def setNumRDDs(numRDDs: Int): this.type = {
    this.numRDDs = numRDDs
    this
  }
  def setMaxAdjusting(maxAdjusting: Int): this.type = {
    this.maxAdjusting = maxAdjusting
    this
  }
  def setLocal(local: Boolean): this.type = {
    this.local = local
    this
  }


  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000

  private def createExpTable(): Array[Float] = {
    val expTable = new Array[Float](EXP_TABLE_SIZE)
    var i = 0
    while (i < EXP_TABLE_SIZE) {
      val tmp = math.exp((2.0 * i / EXP_TABLE_SIZE - 1.0) * MAX_EXP)
      expTable(i) = (tmp / (tmp + 1.0)).toFloat
      i += 1
    }
    expTable
  }
  private def createNegTable(): Array[Int] = {
    val table = new Array[Int](TABEL_SIZE)
    var trainWordsPow = 0.0
    for (a <- 0 to vocabSize - 1)
      trainWordsPow += Math.pow(vocab(a).cn, POWER)
    var i = 0
    var d1 = Math.pow(vocab(i).cn, POWER) / trainWordsPow
    for (a <- 0 to TABEL_SIZE - 1) {
      table(a) = i
      if (a * 1.0 / TABEL_SIZE > d1) {
        i += 1
        d1 += Math.pow(vocab(i).cn, POWER) / trainWordsPow
      }
      if (i >= vocabSize)
        i = vocabSize - 1
    }
    table
  }

  private var vocabSize = 0
  private var vocab: Array[VocabWord] = null
  private var vocabHash = mutable.HashMap.empty[String, Int]
  private var senseTable: Array[Int] = null
  private var totalWords = 0
  private var sentences: RDD[Array[Int]] = null
  private var totalSentences = 0
  private var syn0: Array[Array[Array[Float]]] = null
  private var syn1: Array[Array[Array[Float]]] = null
  private var learned = false
  private var multiSense = false

  def TrainOneSense(input: RDD[String], outputPath: String): Unit = {
    val startTime = currentTime
    var oldTime = startTime

    util.Random.setSeed(seed)

    multiSense = false

    println("learn vocabulary ... ...  ...")
    learnVocab(input)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("create Sense Table ... ... ...")
    createOneSenseTable
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("make sentences ... ... ...")
    makeSentences(input)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("initialize parameters ... ... ...")
    initSynRandomly
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("train (local version) ... ... ...")
    train
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("wrote to file ... ... ...")
    writeToFile(outputPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("total time:"+(currentTime-startTime)/1000.0)
  }

  def TrainTwoSenses(input: RDD[String], synPath: String, outputPath: String): Unit = {
    val startTime = currentTime
    var oldTime = startTime

    multiSense = true

    println("learn vocabulary ... ...  ...")
    learnVocab(input)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("create Sense Table ... ... ...")
    createTwoSensesTable(1000)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("make sentences ... ... ...")
    makeSentences(input)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("initialize parameters ... ... ...")
    initSynFromFile(synPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("train (local version) ... ... ...")
    train
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("wrote to file ... ... ...")
    writeToFile(outputPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("total time:"+(currentTime-startTime)/1000.0)
  }

  def TrainMultipleSenses(): Unit = {

  }

  private def learnVocab(input: RDD[String]): Unit = {
    val words = input.map(line => line.split(" ").array).flatMap(x=>x).filter(x=>x.size>0).map{x=>
      var begin = 0;
      var end = x.size-1;
      while(begin <= end && !x(begin).isLetter)
        begin+=1
      while(begin <= end && !x(end).isLetter)
        end-=1
      x.substring(begin,end+1)
    }.map(x=>x.toLowerCase).filter(x=>x.size>0)

    require(words != null, "words RDD is null. You may need to check if loading data correctly.d")
    vocab = words.map(w => (w, 1))
      .reduceByKey(_ + _)
      .map(x => VocabWord(
        x._1,
        x._2))
      .filter(_.cn >= minCount)
      .collect()
      .sortWith((a, b) => a.cn > b.cn)
    vocabSize = vocab.length
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check " +
      "the setting of minCount, which could be large enough to remove all your words in sentences.")

    for (a <- 0 to vocabSize - 1) {
      vocabHash += vocab(a).word -> a
      totalWords += vocab(a).cn
    }

    println("vocabSize = " + vocabSize)
    println("totalWords = " + totalWords)
  }

  //one sense
  private def createOneSenseTable(): Unit = {
    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    for (a <- 0 to vocabSize - 1) {
      senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 30)
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }
  }

  //two senses
  private def createTwoSensesTable(count: Int): Unit = {
    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    for (a <- 0 to vocabSize - 1) {
      if (vocab(a).word == "so")
        senseTable(a) = 2
      else
        senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 30)
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
      if (vocab(a).word == "so")
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }
  }

  private def makeSentences(input: RDD[String]): Unit = {
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary correctly.")

    val sc = input.context
    val bcVacabHash = sc.broadcast(vocabHash)
    val bcNumSensesTable = sc.broadcast(senseTable)

    sentences = input.map(line => line.split(" ").array).map{sentence=>
      sentence.filter(x=>x.size>0).map{x=>
        var begin = 0;
        var end = x.size-1;
        while(begin <= end && !x(begin).isLetter)
          begin+=1
        while(begin <= end && !x(end).isLetter)
          end-=1
        x.substring(begin,end+1)
      }.map(x=>x.toLowerCase).filter(x=>x.size>0&&bcVacabHash.value.get(x).nonEmpty).map{x=>
        val word = bcVacabHash.value.get(x).get
        word*100+util.Random.nextInt(bcNumSensesTable.value(word))
      }
    }.cache()

    totalSentences = sentences.count().toInt
    println("totalSentences = " + totalSentences)
  }

  //initialize from normal skip-gram model
  private def initSynFromFile(synPath: String): Unit = {


    val syn0Old = Source.fromFile(synPath + "/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s => s.toFloat).toArray
    val syn1Old = Source.fromFile(synPath + "/syn1.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s => s.toFloat).toArray

    require(vectorSize * vocabSize == syn0Old.size, "syn0.size should be equal to vectorSize*vocabSize. You may need to check if vectorSize or vocabSize or syn0 file is correct.")
    require(vectorSize * vocabSize == syn1Old.size, "syn1.size should be equal to vectorSize*vocabSize. You may need to check if vectorSize or vocabSize or syn1 file is correct.")

    syn0 = new Array[Array[Array[Float]]](vocabSize)
    syn1 = new Array[Array[Array[Float]]](vocabSize)

    for (word <- 0 to vocabSize - 1) {

      syn0(word) = new Array[Array[Float]](senseTable(word))
      syn1(word) = new Array[Array[Float]](senseTable(word))


      for (sense <- 0 to senseTable(word) - 1) {

        syn0(word)(sense) = new Array[Float](vectorSize)
        syn1(word)(sense) = new Array[Float](vectorSize)
        for (i <- 0 to vectorSize - 1) {
          syn0(word)(sense)(i) = syn0Old(word * vectorSize + i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
          syn1(word)(sense)(i) = syn1Old(word * vectorSize + i) + (util.Random.nextFloat() - 0.5f) * VARIANCE
        }
      }
    }

    this.vectorSize = vectorSize
  }

  //initialize randomly
  private def initSynRandomly: Unit = {



    syn0 = new Array[Array[Array[Float]]](vocabSize)
    syn1 = new Array[Array[Array[Float]]](vocabSize)

    for (word <- 0 to vocabSize - 1) {

      syn0(word) = new Array[Array[Float]](senseTable(word))
      syn1(word) = new Array[Array[Float]](senseTable(word))

      for (sense <- 0 to senseTable(word) - 1) {

        syn0(word)(sense) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
        syn1(word)(sense) = new Array[Float](vectorSize)
      }
    }

    this.vectorSize = vectorSize
  }

  private def train: Unit = {
    require(syn0 != null, "syn0 should not be null. You may need to check if initializing parameters correctly.")
    require(syn1 != null, "syn1 should not be null. You may need to check if initializing parameters correctly.")

    val sc = sentences.context
    println(sc.defaultParallelism + "   " + sc.master)

    val sentencesSplit = sentences.randomSplit(new Array[Double](numRDDs).map(x=>x+1.0))

    var totalWordCount = 0
    val totalTrainWords = totalWords*iterations
    val totalIterations = iterations * numRDDs

    val bcSyn0 = sc.broadcast(syn0)
    val bcSyn1 = sc.broadcast(syn1)
    val bcExpTable = sc.broadcast(createExpTable)
    val bcNegTable = sc.broadcast(createNegTable)
    val bcSenseTable = sc.broadcast(senseTable)

    for (k <- 1 to totalIterations) {

      val indexRDD = (k-1) % numRDDs
      println("iteration = " + k + "   indexRDD = " + indexRDD)

      //adjust sense assignment
      if (multiSense)
        sentencesSplit(indexRDD) = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
          util.Random.setSeed(seed*idx+k)
          val syn0 = bcSyn0.value
          val syn1 = bcSyn1.value
          val senseTable = bcSenseTable.value
          val expTable = bcExpTable.value
          val negTable = bcNegTable.value

          var sumT = 0

          val newIter = mutable.MutableList[Array[Int]]()
          for (sentence <- iter) {
            var t = 1
            while (adjustSentence(sentence,expTable,negTable,senseTable,syn0,syn1) && t < maxAdjusting)
              t+=1
            newIter+=sentence
            sumT += t
            if (sentence.contains(vocabHash.get("so").get*100) || sentence.contains(vocabHash.get("so").get*100+1)) {
              print("partition "+ idx+",  sentence: ")
              for (element <- sentence) {
                print(vocab(element / 100).word)
                if (vocab(element / 100).word == "so")
                  print("(" + element % 100 + ")")
                print(" ")
              }
              println()
            }
          }

          //println("total sentence iterations:"+sumT+ ",   number of sentences:"+newIter.size + ",   iterations per sentence:" + sumT*1.0/newIter.size)
          newIter.toIterator
        }.cache()


      val accWordCount = sc.accumulator(0)
      val numPartitions = sc.defaultParallelism


      //learn syn0 and syn1
      val tmpRDD = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
        util.Random.setSeed(seed*idx+k)
        val syn0 = bcSyn0.value
        val syn1 = bcSyn1.value
        val senseTable = bcSenseTable.value
        val expTable = bcExpTable.value
        val negTable = bcNegTable.value
        val syn0Modify = new mutable.HashMap[Int,Boolean]
        val syn1Modify = new mutable.HashMap[Int,Boolean]

        var startTime = currentTime
        var lastWordCount = 0
        var wordCount = 0
        var alpha = learningRate * (1 - totalWordCount * 1.0f / totalTrainWords )
        if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f

        for (sentence <- iter) {
          if (wordCount - lastWordCount > 10000) {
            var alpha = learningRate * (1 - (totalWordCount*1.0+wordCount*numPartitions) / totalTrainWords )
            if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f
            //println("partition "+ idx+ ",  wordCount = " + (totalWordCount+wordCount*numPartitions) + "/" +totalTrainWords+ ", "+((wordCount-lastWordCount)*1000/(currentTime-startTime))+" words per second"+", alpha = " + alpha)
            lastWordCount = wordCount
            startTime = currentTime
          }

          for (pos <- 0 to sentence.size-1) {
            val w = sentence(pos)
            val neighbors = getNeighbors(sentence, pos, window)
            val negSamples = getNegSamples(negTable,senseTable)
            learn(w,neighbors,negSamples,syn0,syn1,syn0Modify,syn1Modify,expTable,alpha)
          }

          wordCount += sentence.size
        }

        accWordCount += wordCount


        val synIter = mutable.MutableList[(Int,(Array[Float],Int))]()
        for ((index,nothing)<-syn0Modify)
          synIter += index->(syn0(index/100)(index%100)->1)
        for ((index,nothing)<-syn1Modify)
          synIter += (index+vocabSize*100)->(syn1(index/100)(index%100)->1)

        println("partition "+ idx+ ",  synIter.size = " + synIter.size)

        synIter.toIterator

      }.cache()


      //update syn0 and syn1
      val synUpdate = tmpRDD.reduceByKey{(a,b)=>
        blas.saxpy(vectorSize, 1.0f, b._1, 1, a._1, 1)
        (a._1,b._2+a._2)
      }.collect()


      if (local) {
        tmpRDD.count()
      }
      else {
        val zero = new Array[Float](vectorSize)
        for ((index, (update,count)) <- synUpdate) {
          if (index / 100 < vocabSize) {
            blas.scopy(vectorSize, zero, 1, syn0(index / 100)(index % 100), 1)
            blas.saxpy(vectorSize, 1.0f / count, update, 1, syn0(index / 100)(index % 100), 1)
          }
          else {
            blas.scopy(vectorSize, zero, 1, syn1(index / 100-vocabSize)(index % 100), 1)
            blas.saxpy(vectorSize, 1.0f / count, update, 1, syn1(index / 100-vocabSize)(index % 100), 1)
          }
        }
      }

      totalWordCount += accWordCount.value

      println("syn0(0)(0)(0)=" + syn0(0)(0)(0))
    }

    learned = true
  }

  private def adjustSentence(sentence: Array[Int], expTable: Array[Float], negTable: Array[Int], senseTable: Array[Int], syn0: Array[Array[Array[Float]]], syn1: Array[Array[Array[Float]]]): Boolean = {
    var change = false

    for (pos <- 0 to sentence.size - 1) {

      val negSamples = getNegSamples(negTable, senseTable)
      val neighbors = getNeighbors(sentence, pos, window)
      val word = sentence(pos) / 100

      //select best score
      var bestSense = -1  //there is no best sense
      var bestScore = 0.0
      for (sense <- 0 to senseTable(word)-1) {
        val w = word*100+sense
        val score = getScore(w,neighbors,negSamples,syn0,syn1,expTable)
        if (bestSense == -1 || score > bestScore) {
          bestScore = score
          bestSense = sense
        }
      }

      if (word*100+bestSense != sentence(pos))
        change = true
      sentence(pos) = word*100+bestSense
    }
    change
  }

  private def getNegSamples(negTable: Array[Int], senseTable: Array[Int]): Array[Int] = {
    val negSamples = new Array[Int](negative)
    val tableSize = negTable.size

    for (i <- 0 to negative - 1) {

      negSamples(i) = negTable(Math.abs(util.Random.nextLong() % tableSize).toInt)
      if (negSamples(i) <= 0)
        negSamples(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt

      //add sense information (assign sense randomly)
      negSamples(i) = negSamples(i) * 100 + util.Random.nextInt(senseTable(negSamples(i)))
    }
    negSamples
  }

  private def getNeighbors(sentence: Array[Int], posW: Int, window: Int): Array[Int] = {
    val neighbors = ArrayBuilder.make[Int]
    for (p <- posW - window + 1 to posW + window - 1)
      if (p >= 0 && p < sentence.size && p != posW)
        neighbors += sentence(p)
    neighbors.result()
  }

  private def getScore(w: Int, neighbors: Array[Int], negSamples: Array[Int], syn0: Array[Array[Array[Float]]], syn1: Array[Array[Array[Float]]], expTable: Array[Float]): Double = {
    var score = 0.0
    for (u <- neighbors) {
      score += math.log(activeFunction(syn1(u/100)(u%100), syn0(w/100)(w%100), expTable))
      for (z <- negSamples)
        if (z/100 != w/100)
          score += math.log(1 - activeFunction(syn1(u/100)(u%100), syn0(z/100)(z%100), expTable))
    }
    score
  }

  private def activeFunction(v0: Array[Float], v1: Array[Float], expTable: Array[Float]): Double = {
    val vectorSize = v0.length
    var f = blas.sdot(vectorSize, v0, 1, v1, 1)
    if (f > MAX_EXP)
      f = expTable(expTable.length - 1)
    else if (f < -MAX_EXP)
      f = expTable(0)
    else {
      val ind = ((f + MAX_EXP) * (expTable.size / MAX_EXP / 2.0)).toInt
      f = expTable(ind)
    }
    f
  }

  private def learn(w: Int, neighbors: Array[Int], negSamples: Array[Int], syn0: Array[Array[Array[Float]]], syn1: Array[Array[Array[Float]]], syn0Modify: mutable.HashMap[Int,Boolean], syn1Modify: mutable.HashMap[Int,Boolean],expTable: Array[Float],alpha:Float): Unit = {
    syn0Modify.put(w,true)
    for (u <- neighbors) {
      val neu0e = new Array[Float](vectorSize)
      val g = (1 - activeFunction(syn1(u/100)(u%100), syn0(w/100)(w%100), expTable)).toFloat *alpha
      blas.saxpy(vectorSize, g, syn0(w/100)(w%100), 1, neu0e, 1)
      blas.saxpy(vectorSize, g, syn1(u/100)(u%100), 1, syn0(w/100)(w%100), 1)

      syn1Modify.put(u,true)

      for (z <- negSamples)
        if (z/100 != w/100) {
          val g = (-activeFunction(syn1(u/100)(u%100), syn0(z/100)(z%100), expTable)).toFloat *alpha
          blas.saxpy(vectorSize, g, syn0(z/100)(z%100), 1, neu0e, 1)
          blas.saxpy(vectorSize, g, syn1(u/100)(u%100), 1, syn0(z/100)(z%100), 1)

          syn0Modify.put(z,true)
        }

      blas.saxpy(vectorSize, g, neu0e, 1, syn1(u/100)(u%100), 1)
    }

  }

  private def writeToFile(outputPath: String): Unit = {
    require(learned, "parameters need to be learned. You should learn parameters.")

    val file1 = new PrintWriter(new File(outputPath + "/wordIndex.txt"))
    val file2 = new PrintWriter(new File(outputPath + "/syn0.txt"))
    val file3 = new PrintWriter(new File(outputPath + "/syn1.txt"))
    val file4 = new PrintWriter(new File(outputPath + "/vectors.txt"))
    val wordIndex = vocabHash.toArray.sortWith((a, b) => a._2 < b._2)

    file4.write(vocabSize+" "+vectorSize+"\n")
    for ((wordString, word) <- wordIndex) {
      file4.write(wordString)
      for (sense <- 0 to senseTable(word) - 1) {
        file1.write(wordString + "_" + sense + "\n")
        //println(wordString + "_" + sense + " " + word)
        for (i <- 0 to vectorSize - 1)
          file4.write(" "+syn0(word)(sense)(i))
        file4.write("\n")
      }
    }
    for (word <- 0 to vocabSize - 1) {
      for (sense <- 0 to senseTable(word) - 1) {
        for (i <- 0 to vectorSize - 1) {
          file2.write(syn0(word)(sense)(i) + " ")
          file3.write(syn1(word)(sense)(i) + " ")
        }
        file2.write("\n")
        file3.write("\n")
      }
    }

    file1.close()
    file2.close()
    file3.close()
    file4.close()
  }

}

object SenseAssignment {

  def loadModelSenses(path: String): Word2VecModel = {
    val wordIndex = collection.mutable.Map[String, Int]()
    var index = 0
    for (word <- Source.fromFile(path+"/wordIndex.txt").getLines()) {
      wordIndex.put(word, index)
      index += 1
    }
    val wordVectors = Source.fromFile(path+"/syn0.txt").getLines().map(line => line.split(" ").toSeq).flatten.map(s=>s.toFloat).toArray
    new Word2VecModel(wordIndex.toMap, wordVectors)
  }

}
