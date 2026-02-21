// export_batch.sc
// SINGLE SESSION EXPORT (FAST VERSION)

import java.io.File
import java.io.PrintWriter
import scala.collection.mutable

import io.shiftleft.semanticcpg.language.*
import io.joern.dataflowengineoss.language.*

@main def exportBatch(cpgDir: String, outRoot: String): Unit = {

  val bins =
    new File(cpgDir)
      .listFiles()
      .filter(_.getName.endsWith(".bin"))
      .sorted

  println(s"[+] Found ${bins.length} CPGs")

  bins.foreach { bin =>

    println(s"\n[Export] ${bin.getName}")

    // --------------------------------------
    // LOAD CPG (reuse same Joern session)
    // --------------------------------------
    importCpg(bin.getAbsolutePath)

    println("[+] Ensuring dataflow overlay...")
    // safe — skips if already exists
    run.ossdataflow

    val outputDir =
      new File(outRoot, bin.getName.stripSuffix(".bin"))

    if (!outputDir.exists()) outputDir.mkdirs()

    // --------------------------------------
    // EXPORT
    // --------------------------------------
    cpg.method.filterNot(_.isExternal).l.foreach { m =>

      val nodes = mutable.Set[Long]()
      val edges = mutable.ListBuffer[(Long, Long, String)]()

      val methodNodes = m.ast.l

      // CFG
      methodNodes.foreach { src =>
        src._cfgOut.l.foreach { dst =>
          nodes += src.id; nodes += dst.id
          edges += ((src.id, dst.id, "CFG"))
        }
      }

      // CDG
      methodNodes.foreach { src =>
        src._cdgOut.l.foreach { dst =>
          nodes += src.id; nodes += dst.id
          edges += ((src.id, dst.id, "DOMINATE"))
        }
      }

      // POST DOM
      methodNodes.foreach { src =>
        src._cdgIn.l.foreach { dst =>
          nodes += src.id; nodes += dst.id
          edges += ((src.id, dst.id, "POST_DOMINATE"))
        }
      }

      // TRUE REACHING_DEF
      methodNodes.foreach { src =>
        val it = src.outE("REACHING_DEF")
        while (it.hasNext) {
          val e = it.next()
          val dst = e.inNode()
          nodes += src.id; nodes += dst.id
          edges += ((src.id, dst.id, "REACHING_DEF"))
        }
      }

      // --------------------------------------
      // WRITE DOT
      // --------------------------------------
      val nodeMap =
        nodes.toList
          .flatMap(id => Option(cpg.graph.node(id)).map(n => id -> n))
          .toMap

      val file =
        new File(outputDir, s"${m.name}_${m.id}.dot")

      val writer = new PrintWriter(file)

      writer.println("digraph G {")

      nodeMap.foreach { case (id, n) =>
        val code =
          Option(n.property("CODE"))
            .map(_.toString.replace("\"", "'"))
            .getOrElse("")

        val typ = Option(n.label).getOrElse("UNKNOWN")

        val line =
          Option(n.property("LINE_NUMBER"))
            .map(_.toString)
            .getOrElse("-1")

        writer.println(
          s"""$id [label="$code", code="$code", type="$typ", line_number="$line"];"""
        )
      }

      edges.foreach { case (s, d, t) =>
        writer.println(s"""$s -> $d [type="$t"];""")
      }

      writer.println("}")
      writer.close()
    }

    // --------------------------------------
    // IMPORTANT: CLOSE PROJECT
    // --------------------------------------
    close
    System.gc()
  }

  println("\n✅ Batch export finished.")
}