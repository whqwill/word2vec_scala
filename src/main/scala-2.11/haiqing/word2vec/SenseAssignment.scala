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
  private var epoch = 2
  private var window = 5
  private var negative = 5
  private var learningRate = 0.025f
  private var seed = 42l
  private var numRDDs = 5
  private var maxAdjusting = 10
  private var local = true

  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }
  def setEpoch(epoch: Int): this.type = {
    this.epoch = epoch
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
  private val ALPHA = 0.8
  private val U = 5
  private val ENCODE = 100

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

    println("initialize syn0 and syn1 ... ... ...")
    initSynRandomly
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("make sentences ... ... ...")
    val sentenceRDD = makeSentences(input)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("train (local version) ... ... ...")
    train(sentenceRDD)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("wrote to file ... ... ...")
    writeToFile(outputPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("total time:"+(currentTime-startTime)/1000.0)
  }

  def TrainTwoSenses(input: RDD[String], threshold: Int, synPath: String, outputPath: String): Unit = {
    val startTime = currentTime
    var oldTime = startTime

    multiSense = true

    println("learn vocabulary ... ...  ...")
    learnVocab(input)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("create Sense Table ... ... ...")
    createTwoSensesTable(threshold)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("initialize syn0 and syn1 ... ... ...")
    initSynFromFile(synPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
    oldTime = currentTime

    println("make sentences ... ... ...")
    val sentenceRDD = makeSentences(input)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("train (local version) ... ... ...")
    train(sentenceRDD)
    println("time:"+(currentTime-oldTime)/1000.0)
    oldTime = currentTime

    println("wrote to file ... ... ...")
    writeToFile(outputPath)
    println("time:" + (currentTime - oldTime) / 1000.0)
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
    //remove the beginning and end non-letter, and then transform to lowercase letter
    val words = input.map(line => line.split(" ").array).flatMap(x=>x).filter(x=>x.size>0).map{x=>
      var begin = 0;
      var end = x.size-1;
      while(begin <= end && !x(begin).isLetter)
        begin+=1
      while(begin <= end && !x(end).isLetter)
        end-=1
      x.substring(begin,end+1)
    }.map(x=>x.toLowerCase).filter(x=>x.size>0)

    require(words != null, "words RDD is null. You may need to check if loading data correctly.")
    //build vocabulary with count of each word, and remove the infrequent words
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
      if (vocab(a).word == "apple")
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }
  }

  //two senses
  private def createTwoSensesTable(count: Int): Unit = {
    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    for (a <- 0 to vocabSize - 1) {
      if (vocab(a).word == "apple")
        senseTable(a) = 2
      else
        senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 30)
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
      if (vocab(a).word == "apple")
        println(vocab(a).toString + "  numSenses:" + senseTable(a))
    }
  }

  private def makeSentences(input: RDD[String]): RDD[Array[Int]] = {
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary correctly.")

    val sc = input.context
    val bcVacabHash = sc.broadcast(vocabHash)
    val bcNumSensesTable = sc.broadcast(senseTable)

    val sentenceRDD = input.map(line => line.split(" ").array).map{sentence=>
      sentence.filter(x=>x.size>0).map{x=>
        var begin = 0
        var end = x.size-1
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

    totalSentences = sentenceRDD.count().toInt
    println("totalSentences = " + totalSentences)
    sentenceRDD
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

/*
        if (vocab(word).word=="apple") {
          println("!!!!!")
          syn0(word)(sense) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
          syn1(word)(sense) = new Array[Float](vectorSize)
        }
*/
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
        //syn1(word)(sense) = Array.fill[Float](vectorSize)((util.Random.nextFloat() - 0.5f) / vectorSize)
        syn1(word)(sense) = new Array[Float](vectorSize)
      }
    }

    this.vectorSize = vectorSize
  }

  private def train(sentenceRDD: RDD[Array[Int]]): Unit = {
    require(syn0 != null, "syn0 should not be null. You may need to check if initializing parameters correctly.")
    require(syn1 != null, "syn1 should not be null. You may need to check if initializing parameters correctly.")

    val sc = sentenceRDD.context
    println(sc.defaultParallelism + "   " + sc.master)

    val sentencesSplit = sentenceRDD.randomSplit(new Array[Double](numRDDs).map(x=>x+1.0))

    println("sentencesSplit(0)"+sentencesSplit(0).count())

    if (false) {
      println("!!!!!!!!!!!!!!!!!!!!!!!!!!~~~~~~~~~~~~~~~~~!!!!!!!!!!!")
      sentencesSplit(0) = sc.textFile("new00/new1.txt", sc.defaultParallelism).map(line => line.split(" ").array).map { sentence =>
        sentence.filter(x => x.size > 0).map { x =>
          var begin = 0;
          var end = x.size - 1;
          while (begin <= end && !x(begin).isLetter)
            begin += 1
          while (begin <= end && !x(end).isLetter)
            end -= 1
          x.substring(begin, end + 1)
        }.map(x => x.toLowerCase).filter(x => x.size > 0 && vocabHash.get(x).nonEmpty).map { x =>
          val word = vocabHash.get(x).get
          word * 100 + util.Random.nextInt(senseTable(word))
        }
      }.cache()

      println("sentencesSplit(0)"+sentencesSplit(0).count())

      sentencesSplit(1) = sc.textFile("new00/new2.txt", sc.defaultParallelism).map(line => line.split(" ").array).map { sentence =>
        sentence.filter(x => x.size > 0).map { x =>
          var begin = 0;
          var end = x.size - 1;
          while (begin <= end && !x(begin).isLetter)
            begin += 1
          while (begin <= end && !x(end).isLetter)
            end -= 1
          x.substring(begin, end + 1)
        }.map(x => x.toLowerCase).filter(x => x.size > 0 && vocabHash.get(x).nonEmpty).map { x =>
          val word = vocabHash.get(x).get
          word * 100 + util.Random.nextInt(senseTable(word))
        }
      }.cache()
      println("sentencesSplit(1)"+sentencesSplit(1).count())
    }

    var totalWordCount = 0
    val totalTrainWords = totalWords*epoch
    val iterations = epoch * numRDDs

    val bcSyn0 = sc.broadcast(syn0)
    val bcSyn1 = sc.broadcast(syn1)
    val bcExpTable = sc.broadcast(createExpTable)
    val bcNegTable = sc.broadcast(createNegTable)
    val bcSenseTable = sc.broadcast(senseTable)

    val syn0Modify = new Array[Array[Int]](vocabSize)
    val syn1Modify = new Array[Array[Int]](vocabSize)
    val senseCount = new Array[Array[Int]](vocabSize)
    val senseCountAdd = new Array[Array[Int]](vocabSize)
    for (w <- 0 to vocabSize-1) {
      syn0Modify(w) = new Array[Int](senseTable(w))
      syn1Modify(w) = new Array[Int](senseTable(w))
      senseCount(w) = new Array[Int](senseTable(w))
      senseCountAdd(w) = new Array[Int](senseTable(w))
    }
    val bcSyn0Modify = sc.broadcast(syn0Modify)
    val bcSyn1Modify = sc.broadcast(syn1Modify)

    for (k <- 1 to iterations) {

      val indexRDD = (k-1) % numRDDs
      println("iteration = " + k + "   indexRDD = " + indexRDD)

      val bcSenseCountAdd = sc.broadcast(senseCountAdd)

      //adjust sense assignment
      sentencesSplit(indexRDD) = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
        util.Random.setSeed(seed*idx+k)
        val F = new IterationFunctions
        F.syn0 = bcSyn0.value
        F.syn1 = bcSyn1.value
        F.negative = negative
        F.vocabSize = vocabSize
        F.vectorSize = vectorSize
        F.window = window
        F.ENCODE = ENCODE
        F.MAX_EXP = MAX_EXP
        F.expTable = bcExpTable.value
        F.negTable = bcNegTable.value
        F.senseCountAdd = bcSenseCountAdd.value
        F.senseTable = bcSenseTable.value

        var sumT = 0
        val newIter = mutable.MutableList[Array[Int]]()
        for (sentence <- iter) {
          var t = 1
          while (F.adjustSentence(sentence) && t < maxAdjusting)
            t+=1
          newIter+=sentence
          sumT += t
          F.addSenseCount(sentence)
        }

        println("total sentence iterations:"+sumT+ ",   number of sentences:"+newIter.size + ",   iterations per sentence:" + sumT*1.0/newIter.size)
        newIter.toIterator
      }.cache()
      sentencesSplit(indexRDD).count()

      //update senseCount
      for (word <- 0 to vocabSize-1)
        for (sense <- 0 to senseTable(word)-1) {
          if (word == 1115) {
            var a = 0
            a = 1
          }
          senseCount(word)(sense) = (senseCount(word)(sense)*(1-ALPHA)+senseCountAdd(word)(sense)*ALPHA).toInt
          senseCountAdd(word)(sense) = 0
        }

      if (vocabHash.get("apple").nonEmpty && local) {
        //choose smaller number of sense to print out
        var smallerSense = 0
        var num = totalWords
        for (sense <- 0 to senseTable(vocabHash("apple")) - 1) {
          println("apple" + sense + ": " + senseCount(vocabHash("apple"))(sense))
          if (senseCount(vocabHash("apple"))(sense) < num) {
            num = senseCount(vocabHash("apple"))(sense)
            smallerSense = sense
          }
        }
        //print out the information of sense assignment of "apple"
        sentencesSplit(indexRDD).foreach { sentence =>
          for (pos <- 0 to sentence.size - 1)
            if (vocab(sentence(pos) / 100).word == "apple" && sentence(pos) % 100 == smallerSense) {
              println()
              for (j <- pos - window + 1 to pos + window - 1)
                if (j >= 0 && j < sentence.size) {
                  print(vocab(sentence(j) / 100).word)
                  if (j == pos)
                    print("(" + sentence(j) % 100 + ")")
                  print(" ")
                }
              println()
            }
        }
      }

      val accWordCount = sc.accumulator(0)
      val numPartitions = sc.defaultParallelism
      val bcSenseCount = sc.broadcast(senseCount)

      //learn syn0 and syn1
      val tmpRDD = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
        util.Random.setSeed(seed*idx+k)
        val F = new IterationFunctions
        F.syn0 = bcSyn0.value
        F.syn1 = bcSyn1.value
        F.U = U
        F.window = window
        F.vocabSize = vocabSize
        F.vectorSize = vectorSize
        F.negative = negative
        F.ENCODE = ENCODE
        F.MAX_EXP = MAX_EXP
        F.expTable = bcExpTable.value
        F.negTable = bcNegTable.value
        F.senseCount = bcSenseCount.value
        F.senseTable = bcSenseTable.value
        val syn0Modify = bcSyn0Modify.value
        val syn1Modify = bcSyn1Modify.value

        var startTime = currentTime
        var lastWordCount = 0
        var wordCount = 0
        var alpha = learningRate * (1 - totalWordCount * 1.0f / totalTrainWords )
        if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f

        for (sentence <- iter) {
          if (wordCount - lastWordCount > 10000) {
            var alpha = learningRate * (1 - (totalWordCount*1.0+wordCount*numPartitions) / totalTrainWords )
            if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f
            println("partition "+ idx+ ",  wordCount = " + (totalWordCount+wordCount*numPartitions) + "/" +totalTrainWords+ ", "+((wordCount-lastWordCount)*1000/(currentTime-startTime))+" words per second"+", alpha = " + alpha)
            lastWordCount = wordCount
            startTime = currentTime
          }

          F.alpha = alpha
          F.learnSentence(sentence)
          //about syn0Modify and syn1Modify may be a problem
          wordCount += sentence.size
        }
        accWordCount += wordCount

        val synIter = mutable.MutableList[(Int,(Array[Float],Int))]()
        for (w <- 0 to vocabSize-1)
          for (s <- 0 to senseTable(w)-1) {
            if (syn0Modify(w)(s) > 0)
              synIter += (w*ENCODE+s)->(syn0(w)(s)->syn0Modify(w)(s))
            if (syn1Modify(w)(s) > 0)
              synIter += ((w+vocabSize)*ENCODE+s)->(syn1(w)(s)->syn1Modify(w)(s))
          }

        //println("partition "+ idx+ ",  synIter.size = " + synIter.size)
        println("partition "+ idx+ ",  syn0Modify(0)(0)="+syn0Modify(0)(0))

        synIter.toIterator
      }.cache()

      if (local) {
        tmpRDD.count()
      }
      else {      //update syn0 and syn1
        val synUpdate = tmpRDD.reduceByKey{(a,b)=>
          if (a._2 > b._2)
            a
          else
            b
        }.collect()

        println(synUpdate.size)

        for ((index, (update,count)) <- synUpdate) {
          if (index / ENCODE < vocabSize) {
            blas.saxpy(vectorSize, 1.0f, update, 1, syn0(index / ENCODE)(index % ENCODE), 1)
            syn0Modify(index / ENCODE)(index % 100) = 0
            if (index == 0)
              println("index="+index)
          }
          else {
            blas.saxpy(vectorSize, 1.0f, update, 1, syn1(index / ENCODE-vocabSize)(index % ENCODE), 1)
            syn1Modify(index / ENCODE-vocabSize)(index % ENCODE) = 0
          }
        }
      }
      totalWordCount += accWordCount.value
      println("syn0(0)(0)(0)=" + syn0(0)(0)(0))
      println("syn0Modify(0)(0)=" + syn0Modify(0)(0))
    }

    learned = true
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