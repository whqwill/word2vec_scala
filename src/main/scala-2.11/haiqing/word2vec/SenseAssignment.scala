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
  private var minCountMultiSense = 1000
  private var epoch = 2
  private var window = 5
  private var negative = 5
  private var learningRate = 0.001f
  private var seed = 42l
  private var numRDDs = 5
  private var maxAdjusting = 10
  private var multiSense = 2
  private var local = true
  private var testStep = 5

  def setVectorSize(vectorSize: Int): this.type = {
    this.vectorSize = vectorSize
    this
  }
  def setMinCount(minCount: Int): this.type = {
    this.minCount = minCount
    this
  }
  def setMinCountMultiSense(minCountMultiSense: Int): this.type = {
    this.minCountMultiSense = minCountMultiSense
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
  def setMultiSense(multiSense: Int): this.type = {
    this.multiSense = multiSense
    this
  }
  def setLocal(local: Boolean): this.type = {
    this.local = local
    this
  }
  def setTestStep(testStep: Int): this.type = {
    this.testStep = testStep
    this
  }

  private val EXP_TABLE_SIZE = 1000
  private val MAX_EXP = 6
  private val POWER = 0.75
  private val VARIANCE = 0.01f
  private val TABEL_SIZE = 10000
  private val ALPHA = 0.8
  private val U = 100
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

  def TrainOneSense(input: RDD[String], outputPath: String): Unit = {
    val startTime = currentTime

    util.Random.setSeed(seed)

    learnVocab(input)

    createSenseTable()

    initSynRandomly()

    val sentenceRDD = makeSentences(input)

    train(sentenceRDD)

    writeToFile(outputPath)

    println("total time:"+(currentTime-startTime)/1000.0)
  }

  def TrainMultiSense(input: RDD[String], synPath: String, outputPath: String): Unit = {
    val startTime = currentTime

    util.Random.setSeed(seed)

    learnVocab(input)

    createSenseTable()

    initSynFromFile(synPath)

    val sentenceRDD = makeSentences(input)

    train(sentenceRDD)

    writeToFile(outputPath)

    println("total time:"+(currentTime-startTime)/1000.0)
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

  private def createSenseTable(): Unit = {
    senseTable = new Array[Int](vocabSize)

    println("print some words in vocabulary: ")
    var multiSenseWord = 0
    for (a <- 0 to vocabSize - 1) {
      if (vocab(a).cn > minCountMultiSense) {
        senseTable(a) = multiSense
        multiSenseWord += 1
      }
      else
        senseTable(a) = 1
      if (util.Random.nextInt(vocabSize) < 300 && senseTable(a) == multiSense)
        println(vocab(a).toString + "  sense:" + senseTable(a))
      if (util.Random.nextInt(vocabSize) < 30 && senseTable(a) == 1)
        println(vocab(a).toString + "  sense:" + senseTable(a))
    }
    println("multiSenseWord = "+multiSenseWord+"   oneSenseWord = " + totalWords)
  }

  private def makeSentences(input: RDD[String]): RDD[(Array[Int],Array[Array[Int]])] = {
    require(vocabSize > 0, "The vocabulary size should be > 0. You may need to check if learning vocabulary correctly.")

    val sc = input.context
    val bcVocabHash = sc.broadcast(vocabHash)
    val bcSenseTable = sc.broadcast(senseTable)
    val bcNegTable = sc.broadcast(createNegTable())

    val sentenceRDD = input.map(line => line.split(" ").array).map{sentence=>
      val newSentence = sentence.filter(x=>x.size>0).map{x=>
        var begin = 0
        var end = x.size-1
        while(begin <= end && !x(begin).isLetter)
          begin+=1
        while(begin <= end && !x(end).isLetter)
          end-=1
        x.substring(begin,end+1)
      }.map(x=>x.toLowerCase).filter(x=>x.size>0&&bcVocabHash.value.get(x).nonEmpty).map{x=>
        val word = bcVocabHash.value.get(x).get
        word*100+util.Random.nextInt(bcSenseTable.value(word))
      }
      val sentenceNEG = newSentence.map(w=>getNEG(w,bcNegTable.value))
      (newSentence, sentenceNEG)
    }.cache()

    totalSentences = sentenceRDD.count().toInt
    println("totalSentences = " + totalSentences)
    sentenceRDD
  }

  private def getNEG(w: Int, negTable: Array[Int]): Array[Int] = {
    val negSamples = new Array[Int](negative)
    val tableSize = negTable.size
    for (i <- 0 to negative - 1) {
      negSamples(i) = w
      while (negSamples(i)/ENCODE == w/ENCODE) {
        negSamples(i) = negTable(Math.abs(util.Random.nextLong() % tableSize).toInt)
        if (negSamples(i) <= 0)
          negSamples(i) = (Math.abs(util.Random.nextLong()) % (vocabSize - 1) + 1).toInt
      }
      //add sense information (assign sense randomly)
      negSamples(i) = negSamples(i) * ENCODE + util.Random.nextInt(senseTable(negSamples(i)))
    }
    negSamples
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
  private def initSynRandomly(): Unit = {

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

  private def train(sentenceRDD: RDD[(Array[Int],Array[Array[Int]])]): Unit = {
    require(syn0 != null, "syn0 should not be null. You may need to check if initializing parameters correctly.")
    require(syn1 != null, "syn1 should not be null. You may need to check if initializing parameters correctly.")

    val sc = sentenceRDD.context
    val sentencesSplit = sentenceRDD.randomSplit(new Array[Double](numRDDs).map(x=>x+1.0))
    val numPartitions = sc.defaultParallelism

    var totalWordCount = 0
    val totalTrainWords = totalWords*epoch
    val iterations = epoch * numRDDs

    val bcSyn0 = sc.broadcast(syn0)
    val bcSyn1 = sc.broadcast(syn1)
    val bcExpTable = sc.broadcast(createExpTable)
    val bcSenseTable = sc.broadcast(senseTable)

    //val syn0Modify = new Array[Array[Int]](vocabSize)
    //val syn1Modify = new Array[Array[Int]](vocabSize)
    val senseCount = new Array[Array[Int]](vocabSize)
    val senseCountAdd = new Array[Array[Int]](vocabSize)
    for (w <- 0 to vocabSize-1) {
      //syn0Modify(w) = new Array[Int](senseTable(w))
      //syn1Modify(w) = new Array[Int](senseTable(w))
      senseCount(w) = new Array[Int](senseTable(w))
      senseCountAdd(w) = new Array[Int](senseTable(w))
      for (s <- 0 to senseTable(w)-1)
        senseCount(w)(s) = vocab(w).cn/senseTable(w)
    }
    //val bcSyn0Modify = sc.broadcast(syn0Modify)
    //val bcSyn1Modify = sc.broadcast(syn1Modify)

    //val file = new PrintWriter(new File("./tmp.txt"))

    for (k <- 1 to iterations) {

      val indexRDD = (k-1) % numRDDs

      //initialize senseCountAdd and broadcast senseCountAdd
      for (word <- 0 to vocabSize-1)
        for (sense <- 0 to senseTable(word)-1) {
          senseCountAdd(word)(sense) = 0
        }
      val bcSenseCountAdd = sc.broadcast(senseCountAdd)

      val loss = sc.accumulator(0.0)
      val lossNum = sc.accumulator(0)
      val adjust = sc.accumulator(0)

      println("iteration = " + k + "   indexRDD = " + indexRDD+ " adjust sense assignment ...")
      //adjust sense assignment
      sentencesSplit(indexRDD) = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
        util.Random.setSeed(seed*idx+k)
        val F = new IterationFunctions(window,vectorSize,multiSense,bcSenseCountAdd.value,null,bcExpTable.value,bcSenseTable.value,bcSyn0.value,bcSyn1.value)
        val newIter = mutable.MutableList[(Array[Int],Array[Array[Int]])]()
        for ((sentence,sentenceNEG) <- iter) {
          var t = 1
          F.setSentence(sentence)
          F.setSentenceNEG(sentenceNEG)ope
          while (F.adjustSentence() && t < maxAdjusting)
            t+=1
          newIter+=sentence->sentenceNEG
          adjust += t
          F.addSenseCount()
          loss += F.sentenceLoss()
          lossNum += sentence.size
        }
        newIter.toIterator
      }.cache()
      val sentenceNum = sentencesSplit(indexRDD).count()
      println("Average number of adjustments per sentence: " + adjust.value*1.0/sentenceNum)
      println("Average loss per word: "+loss.value/lossNum.value)

      println("iteration = " + k + "   indexRDD = " + indexRDD+ " update senseCount ...")
      //update senseCount
      for (word <- 0 to vocabSize-1)
        for (sense <- 0 to senseTable(word)-1) {
          //maybe a problem
          senseCount(word)(sense) = (senseCount(word)(sense)*(1-ALPHA)+senseCountAdd(word)(sense)*ALPHA).toInt
        }

      /*
      println("print out senseCountAdd and senseCount ...")
      //print out senseCountAdd and senseCount
      if (multiSense > 1 && local) {
        file.write("Iteration "+k+":\n")
        for (word <- 0 to vocabSize-1)
          if (senseTable(word) == multiSense && (vocab(word).word == "apple" || vocab(word).word == "bank" || vocab(word).word == "bad" || vocab(word).word == "school"))
            for (sense <- 0 to senseTable(word) - 1) {
              println(vocab(word).word + sense + ": +" + senseCountAdd(word)(sense) +"  -> "+ senseCount(word)(sense))
              file.write(vocab(word).word + sense + ": " + senseCount(word)(sense)+"\n")
            }
        file.write("\n")

        //print out the information of sense assignment of "apple"
        /*
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
        }*/
      }
      */

      //wordCount and numPartitions are for updating the alpha(learning rate)
      val accWordCount = sc.accumulator(0)
      //broadcast senseCount
      val bcSenseCount = sc.broadcast(senseCount)

      println("iteration = " + k + "   indexRDD = " + indexRDD + " learn syn0 and syn1 ...")
      //learn syn0 and syn1
      val tmpRDD = sentencesSplit(indexRDD).mapPartitionsWithIndex {(idx,iter)=>
        util.Random.setSeed(seed*idx+k)
        val F = new IterationFunctions(window,vectorSize,multiSense,null,bcSenseCount.value,bcExpTable.value,bcSenseTable.value,bcSyn0.value,bcSyn1.value)

        //update alpha
        var startTime = currentTime
        var lastWordCount = 0
        var wordCount = 0
        var alpha = learningRate * (1 - totalWordCount * 1.0f / totalTrainWords )
        if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f

        for ((sentence,sentenceNEG) <- iter) {
          if (wordCount - lastWordCount > 10000) {
            var alpha = learningRate * (1 - (totalWordCount*1.0+wordCount*numPartitions) / totalTrainWords )
            if (alpha < learningRate * 0.0001f) alpha = learningRate * 0.0001f
            //println("partition "+ idx+ ",  wordCount = " + (totalWordCount+wordCount*numPartitions) + "/" +totalTrainWords+ ", "+((wordCount-lastWordCount)*1000/(currentTime-startTime))+" words per second"+", alpha = " + alpha)
            lastWordCount = wordCount
            startTime = currentTime
          }
          F.setSentence(sentence)
          F.setSentenceNEG(sentenceNEG)
          F.learnSentence(alpha)
          //about syn0Modify and syn1Modify may be a problem
          wordCount += sentence.size
        }
        accWordCount += wordCount

        val synIter = mutable.MutableList[(Int,Array[Float])]()
        for (word <- 0 to vocabSize-1)
          for (sense <- 0 to senseTable(word)-1) {
            //if (syn0Modify(w)(s) > 0)
              synIter += (word*ENCODE+sense)->bcSyn0.value(word)(sense)
            //if (syn1Modify(w)(s) > 0)
              synIter += ((word+vocabSize)*ENCODE+sense)->bcSyn1.value(word)(sense)
          }

        //println("partition "+ idx+ ",  synIter.size = " + synIter.size)
        //println("partition "+ idx+ ",  syn0Modify(0)(0)="+syn0Modify(0)(0))

        //val synIter = mutable.MutableList[(Int,(Array[Float],Int))]()
        synIter.toIterator
      }.cache()

      if (local) {
        tmpRDD.count()
      }
      else {
        //update syn0 and syn1
        val synUpdate = tmpRDD.reduceByKey{(a,b)=>
          blas.saxpy(vectorSize, 1.0f, b, 1, a, 1)
          a
        }.collect()

        println(synUpdate.size)

        for ((index, newSyn) <- synUpdate) {
          if (index / ENCODE < vocabSize) {
            blas.saxpy(vectorSize, 1.0f/numPartitions, newSyn, 1, syn0(index/ENCODE)(index%ENCODE), 1)
            //syn0Modify(index / ENCODE)(index % 100) = 0
            //if (index == 0)
            //  println("index="+index)
          }
          else {
            blas.saxpy(vectorSize, 1.0f/numPartitions, newSyn, 1, syn1(index/ENCODE-vocabSize)(index%ENCODE), 1)
            //syn1Modify(index / ENCODE-vocabSize)(index % ENCODE) = 0
          }
        }

      }
      totalWordCount += accWordCount.value
      println()
      //println("syn0(0)(0)(0)=" + syn0(0)(0)(0))
      //println("syn0Modify(0)(0)=" + syn0Modify(0)(0))
      
    }
    //file.close()
  }

  private def writeToFile(outputPath: String): Unit = {

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