/**
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://www.graphlab.ml.cmu.edu
 *
 */




#ifndef GRAPHLAB_XASYNC_CONSISTENT_ENGINE
#define GRAPHLAB_XASYNC_CONSISTENT_ENGINE

#include <deque>
#include <boost/bind.hpp>

#include <graphlab/scheduler/ischeduler.hpp>
#include <graphlab/scheduler/scheduler_factory.hpp>
#include <graphlab/scheduler/get_message_priority.hpp>
#include <graphlab/vertex_program/ivertex_program.hpp>
#include <graphlab/vertex_program/icontext.hpp>
#include <graphlab/vertex_program/context.hpp>
#include <graphlab/engine/iengine.hpp>
#include <graphlab/engine/execution_status.hpp>
#include <graphlab/options/graphlab_options.hpp>
#include <graphlab/rpc/dc_dist_object.hpp>
#include <graphlab/engine/distributed_chandy_misra.hpp>
#include <graphlab/engine/message_array.hpp>

#include <graphlab/util/tracepoint.hpp>
#include <graphlab/util/memory_info.hpp>
#include <graphlab/util/generics/conditional_addition_wrapper.hpp>
#include <graphlab/rpc/distributed_event_log.hpp>
#include <graphlab/parallel/fiber_group.hpp>
#include <graphlab/parallel/fiber_control.hpp>
#include <graphlab/rpc/fiber_async_consensus.hpp>
#include <graphlab/aggregation/distributed_aggregator.hpp>
#include <graphlab/parallel/fiber_remote_request.hpp>

//=========== include need by sync engine
#include <graphlab/parallel/pthread_tools.hpp>
#include <graphlab/rpc/buffered_exchange.hpp>


#include <graphlab/macros_def.hpp>

//xie insert define
#define X_SYNC 1
#define X_ASYNC 2
#define X_MANUAL 3
#define X_ADAPTIVE 4
#define X_SAMPLE 5

//#define LEAST_EXECUTE_TIME 1


namespace graphlab {


  /**
   * \ingroup engines
   *
   * \brief The asynchronous consistent engine executed vertex programs
   * asynchronously and can ensure mutual exclusion such that adjacent vertices
   * are never executed simultaneously. The default mode is "factorized"
   * consistency in which only individual gathers/applys/
   * scatters are guaranteed to be consistent, but this can be strengthened to
   * provide full mutual exclusion.
   *
   *
   * \tparam VertexProgram
   * The user defined vertex program type which should implement the
   * \ref graphlab::ivertex_program interface.
   *
   * ### Execution Semantics
   *
   * On start() the \ref graphlab::ivertex_program::init function is invoked
   * on all vertex programs in parallel to initialize the vertex program,
   * vertex data, and possibly signal vertices.
   *
   * After which, the engine spawns a collection of threads where each thread
   * individually performs the following tasks:
   * \li Extract a message from the scheduler.
   * \li Perform distributed lock acquisition on the vertex which is supposed
   * to receive the message. The lock system enforces that no neighboring
   * vertex is executing at the same time. The implementation is based
   * on the Chandy-Misra solution to the dining philosophers problem.
   * (Chandy, K.M.; Misra, J. (1984). The Drinking Philosophers Problem.
   *  ACM Trans. Program. Lang. Syst)
   * \li Once lock acquisition is complete,
   *  \ref graphlab::ivertex_program::init is called on the vertex
   * program. As an optimization, any messages sent to this vertex
   * before completion of lock acquisition is merged into original message
   * extracted from the scheduler.
   * \li Execute the gather on the vertex program by invoking
   * the user defined \ref graphlab::ivertex_program::gather function
   * on the edge direction returned by the
   * \ref graphlab::ivertex_program::gather_edges function.  The gather
   * functions can modify edge data but cannot modify the vertex
   * program or vertex data and can be executed on multiple
   * edges in parallel.
   * * \li Execute the apply function on the vertex-program by
   * invoking the user defined \ref graphlab::ivertex_program::apply
   * function passing the sum of the gather functions.  If \ref
   * graphlab::ivertex_program::gather_edges returns no edges then
   * the default gather value is passed to apply.  The apply function
   * can modify the vertex program and vertex data.
   * \li Execute the scatter on the vertex program by invoking
   * the user defined \ref graphlab::ivertex_program::scatter function
   * on the edge direction returned by the
   * \ref graphlab::ivertex_program::scatter_edges function.  The scatter
   * functions can modify edge data but cannot modify the vertex
   * program or vertex data and can be executed on multiple
   * edges in parallel.
   * \li Release all locks acquired in the lock acquisition stage,
   * and repeat until the scheduler is empty.
   *
   * The engine threads multiplexes the above procedure through a secondary
   * internal queue, allowing an arbitrary large number of vertices to
   * begin processing at the same time.
   *
   * ### Construction
   *
   * The asynchronous consistent engine is constructed by passing in a
   * \ref graphlab::distributed_control object which manages coordination
   * between engine threads and a \ref graphlab::distributed_graph object
   * which is the graph on which the engine should be run.  The graph should
   * already be populated and cannot change after the engine is constructed.
   * In the distributed setting all program instances (running on each machine)
   * should construct an instance of the engine at the same time.
   *
   * Computation is initiated by signaling vertices using either
   * \ref graphlab::async_consistent_engine::signal or
   * \ref graphlab::async_consistent_engine::signal_all.  In either case all
   * machines should invoke signal or signal all at the same time.  Finally,
   * computation is initiated by calling the
   * \ref graphlab::async_consistent_engine::start function.
   *
   * ### Example Usage
   *
   * The following is a simple example demonstrating how to use the engine:
   * \code
   * #include <graphlab.hpp>
   *
   * struct vertex_data {
   *   // code
   * };
   * struct edge_data {
   *   // code
   * };
   * typedef graphlab::distributed_graph<vertex_data, edge_data> graph_type;
   * typedef float gather_type;
   * struct pagerank_vprog :
   *   public graphlab::ivertex_program<graph_type, gather_type> {
   *   // code
   * };
   *
   * int main(int argc, char** argv) {
   *   // Initialize control plain using mpi
   *   graphlab::mpi_tools::init(argc, argv);
   *   graphlab::distributed_control dc;
   *   // Parse command line options
   *   graphlab::command_line_options clopts("PageRank algorithm.");
   *   std::string graph_dir;
   *   clopts.attach_option("graph", &graph_dir, graph_dir,
   *                        "The graph file.");
   *   if(!clopts.parse(argc, argv)) {
   *     std::cout << "Error in parsing arguments." << std::endl;
   *     return EXIT_FAILURE;
   *   }
   *   graph_type graph(dc, clopts);
   *   graph.load_structure(graph_dir, "tsv");
   *   graph.finalize();
   *   std::cout << "#vertices: " << graph.num_vertices()
   *             << " #edges:" << graph.num_edges() << std::endl;
   *   graphlab::async_consistent_engine<pagerank_vprog> engine(dc, graph, clopts);
   *   engine.signal_all();
   *   engine.start();
   *   std::cout << "Runtime: " << engine.elapsed_time();
   *   graphlab::mpi_tools::finalize();
   * }
   * \endcode
   *
   * \see graphlab::omni_engine
   * \see graphlab::synchronous_engine
   *
   * <a name=engineopts>Engine Options</a>
   * =========================
   * The asynchronous engine supports several engine options which can
   * be set as command line arguments using \c --engine_opts :
   *
   * \li \b timeout (default: infinity) Maximum time in seconds the engine will
   * run for. The actual runtime may be marginally greater as the engine
   * waits for all threads and processes to flush all active tasks before
   * returning.
   * \li \b factorized (default: true) Set to true to weaken the consistency
   * model to factorized consistency where only individual gather/apply/scatter
   * calls are guaranteed to be locally consistent. Can produce massive
   * increases in throughput at a consistency penalty.
   * \li \b nfibers (default: 3000) Number of fibers to use
   * \li \b stacksize (default: 16384) Stacksize of each fiber.
   */
		template<typename VertexProgram>
		class xadaptive_engine: public iengine<VertexProgram> {
	  
		public:
		  /**
		   * \brief The user defined vertex program type. Equivalent to the
		   * VertexProgram template argument.
		   *
		   * The user defined vertex program type which should implement the
		   * \ref graphlab::ivertex_program interface.
		   */
		  typedef VertexProgram vertex_program_type;
	  
		  /**
		   * \brief The user defined type returned by the gather function.
		   *
		   * The gather type is defined in the \ref graphlab::ivertex_program
		   * interface and is the value returned by the
		   * \ref graphlab::ivertex_program::gather function.  The
		   * gather type must have an <code>operator+=(const gather_type&
		   * other)</code> function and must be \ref sec_serializable.
		   */
		  typedef typename VertexProgram::gather_type gather_type;
	  
		  /**
		   * \brief The user defined message type used to signal neighboring
		   * vertex programs.
		   *
		   * The message type is defined in the \ref graphlab::ivertex_program
		   * interface and used in the call to \ref graphlab::icontext::signal.
		   * The message type must have an
		   * <code>operator+=(const gather_type& other)</code> function and
		   * must be \ref sec_serializable.
		   */
		  typedef typename VertexProgram::message_type message_type;
	  
		  /**
		   * \brief The type of data associated with each vertex in the graph
		   *
		   * The vertex data type must be \ref sec_serializable.
		   */
		  typedef typename VertexProgram::vertex_data_type vertex_data_type;
	  
		  /**
		   * \brief The type of data associated with each edge in the graph
		   *
		   * The edge data type must be \ref sec_serializable.
		   */
		  typedef typename VertexProgram::edge_data_type edge_data_type;
	  
		  /**
		   * \brief The type of graph supported by this vertex program
		   *
		   * See graphlab::distributed_graph
		   */
		  typedef typename VertexProgram::graph_type graph_type;
	  
		   /**
		   * \brief The type used to represent a vertex in the graph.
		   * See \ref graphlab::distributed_graph::vertex_type for details
		   *
		   * The vertex type contains the function
		   * \ref graphlab::distributed_graph::vertex_type::data which
		   * returns a reference to the vertex data as well as other functions
		   * like \ref graphlab::distributed_graph::vertex_type::num_in_edges
		   * which returns the number of in edges.
		   *
		   */
		  typedef typename graph_type::vertex_type			vertex_type;
	  
		  /**
		   * \brief The type used to represent an edge in the graph.
		   * See \ref graphlab::distributed_graph::edge_type for details.
		   *
		   * The edge type contains the function
		   * \ref graphlab::distributed_graph::edge_type::data which returns a
		   * reference to the edge data.  In addition the edge type contains
		   * the function \ref graphlab::distributed_graph::edge_type::source and
		   * \ref graphlab::distributed_graph::edge_type::target.
		   *
		   */
		  typedef typename graph_type::edge_type			edge_type;
	  
		  /**
		   * \brief The type of the callback interface passed by the engine to vertex
		   * programs.	See \ref graphlab::icontext for details.
		   *
		   * The context callback is passed to the vertex program functions and is
		   * used to signal other vertices, get the current iteration, and access
		   * information about the engine.
		   */
		  typedef icontext<graph_type, gather_type, message_type> icontext_type;
	  
		private:
		  // xie insert : this marks the current engine;  X_SYNC  X_ASYNC 
		  int current_engine;
		  int running_mode;
		  
		  /// \internal \brief The base type of all schedulers
		  message_array<message_type> xmessages;		  // xie insert : this is messages of async engine
	  
		  /** \internal
		   * \brief The true type of the callback context interface which
		   * implements icontext. \see graphlab::icontext graphlab::context
		   */
		  typedef context<xadaptive_engine> context_type;
	  
		  // context needs access to internal functions
		  friend class context<xadaptive_engine>;
	  
		  /// \internal \brief The type used to refer to vertices in the local graph
		  typedef typename graph_type::local_vertex_type	local_vertex_type;
		  /// \internal \brief The type used to refer to edges in the local graph
		  typedef typename graph_type::local_edge_type		local_edge_type;
		  /// \internal \brief The type used to refer to vertex IDs in the local graph
		  typedef typename graph_type::lvid_type			lvid_type;
	  
		  /// \internal \brief The type of the current engine instantiation
		  typedef xadaptive_engine<VertexProgram> engine_type;
	  
		  typedef conditional_addition_wrapper<gather_type> conditional_gather_type;
		  
		  /// The RPC interface
		  dc_dist_object<xadaptive_engine<VertexProgram> > rmi;
	  
		  /// A reference to the active graph
		  graph_type& graph;
	  
		  /// A pointer to the lock implementation
		  distributed_chandy_misra<graph_type>* cmlocks;	  
		  
		  /// Per vertex data locks
		  std::vector<simple_spinlock> vertexlocks;
	  
	  
		  /**
		   * \brief This optional vector contains caches of previous gather
		   * contributions for each machine.
		   *
		   * Caching is done locally and therefore a high-degree vertex may
		   * have multiple caches (one per machine).
		   */
		  std::vector<gather_type>	gather_cache;
	  
		  /**
		   * \brief A bit indicating if the local gather for that vertex is
		   * available.
		   */
		  dense_bitset has_cache;
	  
		  bool use_cache;
	  
		  /// Engine threads.
		  fiber_group thrgroup;
	  
		  //! The scheduler
		  ischeduler* scheduler_ptr;
	  
		  typedef typename iengine<VertexProgram>::aggregator_type aggregator_type;
		  aggregator_type aggregator;
	  
		  /// Number of kernel threads
		  size_t ncpus;
		  /// Size of each fiber stack
		  size_t stacksize;
		  /// Number of fibers
		  size_t nfibers;
		  /// set to true if engine is started
		  bool started;
		  /// A pointer to the distributed consensus object
		  fiber_async_consensus* consensus;
	  
		  /**
		   * Used only by the locking subsystem.
		   * to allow the fiber to go to sleep when waiting for the locks to
		   * be ready.
		   */
		  struct vertex_fiber_cm_handle {
			mutex lock;
			bool philosopher_ready;
			size_t fiber_handle;
		  };
		  std::vector<vertex_fiber_cm_handle*> cm_handles;
	  
		  dense_bitset program_running;
		  dense_bitset hasnext;
	  
		  // Various counters.
		  atomic<uint64_t> programs_executed;
	  
		  timer launch_timer;
	  
		  /// Defaults to (-1), defines a timeout
		  size_t timed_termination;
	   
		  /// engine option. Sets to true if factorized consistency is used
		  bool factorized_consistency;
	  
		  bool endgame_mode;
	  
		  /// Time when engine is started
		  float engine_start_time;
	  
		  //xie insert
		  float x_start_time_m;
	  
		  /// True when a force stop is triggered (possibly via a timeout)
		  bool force_stop;
	  
		  graphlab_options opts_copy; // local copy of options to pass to
									  // scheduler construction
	  
		  execution_status::status_enum termination_reason;
	  
		  std::vector<mutex> aggregation_lock;
		  std::vector<std::deque<std::string> >  aggregation_queue;
	  
	  
		  /*
		  *   ======================================================================
		  *   ======================== variable of sync mode =============================
		  */
		  
		  std::vector<double> per_thread_compute_time;
		  
		  /**
		   * \brief The local worker threads used by this engine
		   */
		  thread_pool threads;
		  
		  /**
		   * \brief A thread barrier that is used to control the threads in the
		   * thread pool.
		   */
		  graphlab::barrier thread_barrier;
		  
		  size_t max_iterations;
		  bool has_max_iterations;
		  
		  /**
		   * \brief A snapshot is taken every this number of iterations.
		   * If snapshot_interval == 0, a snapshot is only taken before the first
		   * iteration. If snapshot_interval < 0, no snapshots are taken.
		   */
		  int snapshot_interval;
		  
		  /// \brief The target base name the snapshot is saved in.
		  std::string snapshot_path;
		  
		  /**
		   * \brief A counter that tracks the current iteration number since
		   * start was last invoked.
		   */
		  int iteration_counter;
	  
		  /**
			  * \brief The time in seconds at which the engine started.
			  */
		  float start_time;
	  
		  
		  /**
			  * \brief The timeout time in seconds
			  */
		  float timeout;
		  
		  /**
		   * \brief Schedules all vertices every iteration
		   */
		  bool sched_allv;
	  
		  /**
			  * \brief Used to stop the engine prematurely
			  */
		  bool force_abort;
		  
		  /**
		   * \brief The vertex locks protect access to vertex specific
		   * data-structures including
		   * \ref graphlab::synchronous_engine::gather_accum
		   * and \ref graphlab::synchronous_engine::messages.
		   */
		  std::vector<simple_spinlock> vlocks;
		  
		  
		  /**
		   * \brief The elocks protect individual edges during gather and
		   * scatter.  Technically there is a potential race since gather
		   * and scatter can modify edge values and can overlap.  The edge
		   * lock ensures that only one gather or scatter occurs on an edge
		   * at a time.
		   */
		  std::vector<simple_spinlock> elocks;
	  
		  /**
			   * \brief The vertex programs associated with each vertex on this
			   * machine.
			   */
		  std::vector<vertex_program_type> vertex_programs;
		  
		  /**
		   * \brief Vector of messages associated with each vertex.
		   */
		  std::vector<message_type> messages;
	  
		  /**
		   * \brief Bit indicating whether a message is present for each vertex.
		   */
		  dense_bitset has_message;
		  
		  /**
		   * \brief Gather accumulator used for each master vertex to merge
		   * the result of all the machine specific accumulators (or
		   * caches).
		   *
		   * The gather accumulator can be accessed by multiple threads at
		   * once and therefore must be guarded by a vertex locks in
		   * \ref graphlab::synchronous_engine::vlocks
		   */
		  std::vector<gather_type>	gather_accum;
		  
		  /**
		   * \brief Bit indicating if the gather has accumulator contains any
		   * values.
		   *
		   * While dense bitsets are thread safe the value of this bit must
		   * change concurrently with the
		   * \ref graphlab::synchronous_engine::gather_accum and therefore is
		   * set while holding the lock in
		   * \ref graphlab::synchronous_engine::vlocks.
		   */
		  dense_bitset has_gather_accum;
	  
		  //xie insert:  mark the active node when engine changes.
		  dense_bitset next_mode_active_vertex;
	  
		  //xie insert:  mark the active node in ASYNC engine 
		  //dense_bitset asy_now_active_v;		  //set in async mode for next SYNC
		  dense_bitset asy_start_active_v;	  //set in signal of sync mode for next ASYNC
	  
		  //xie insert: count msg
		  atomic<size_t> each_iteration_nmsg;
		  atomic<size_t> each_iteration_signal;
		  bool	  first_time_start;
		  bool	  local_sync;
		  bool	  stop_async;
		  atomic<uint64_t> exclusive_executed;
		  size_t exclusive_executed_pre;
		  size_t program_executed_pre;
	  
		  //sample mode
		  double sample_start;
		  double throughput;
		  
		  //manual switch
		  long tasknum;
		  size_t switch_iter;
		  
	  
		  //adaptive switch
		  double rate_AvsS;
		  double thro_A;
		  size_t A_Sampled_Iters;
		  double s_c;
		  
		  // threshold in SYNC
		  float  X_S_Increase_Rate;   // -1/100000		  if >1 it keeps increase£¬ if >2 it is 2 exponent
		  size_t X_S_Min_Iters; 	  // 5
		  size_t X_S_Sampled_Iters;   // 10   10day avg line  pick 5
		  //float  X_S_Actived_Rate;  //0.01/machine_num
	  
		  // threshold in ASYNC 
		  float  X_A_Threshold_low;// 0.01
		  float  X_A_Threshold_hig;// 0.015
		  float  X_A_Join_Rate;
		  size_t X_A_Delay_Counter;
		  size_t X_A_Delay;
		  size_t T_SAMPLE;
		  
		  size_t d_join;
		  size_t d_add;
		  size_t d_complete;
		  size_t delay;
		  float avg_line[11];
		  float active[11];
		  size_t startcounter[30];
		  size_t endcounter[30];
		  size_t lastcounter;
		  size_t lastcounters;
		  timer globaltimer;
		  double lastime;
		  double startend;
		  double countoverhead;
		  
		  /**
		   * \brief A bit (for master vertices) indicating if that vertex is active
		   * (received a message on this iteration).
		   */
		  dense_bitset active_superstep;
		  
		  /**
		   * \brief  The number of local vertices (masters) that are active on this
		   * iteration.
		   */
		  atomic<size_t> num_active_vertices;
		  //xie insert
		  atomic<size_t> num_active_mirrors;
		  
		  /**
		   * \brief A bit indicating (for all vertices) whether to
		   * participate in the current minor-step (gather or scatter).
		   */
		  dense_bitset active_minorstep;
		  
		  /**
		   * \brief A counter measuring the number of applys that have been completed
		   */
		  atomic<size_t> completed_applys;
		  
		  
		  /**
		   * \brief The shared counter used coordinate operations between
		   * threads.
		   */
		  atomic<size_t> shared_lvid_counter;
		  
		  
		  /**
		   * \brief The pair type used to synchronize vertex programs across machines.
		   */
		  typedef std::pair<vertex_id_type, vertex_program_type> vid_prog_pair_type;
		  
		  /**
		   * \brief The type of the exchange used to synchronize vertex programs
		   */
		  typedef buffered_exchange<vid_prog_pair_type> vprog_exchange_type;
		  
		  /**
		   * \brief The distributed exchange used to synchronize changes to
		   * vertex programs.
		   */
		  vprog_exchange_type vprog_exchange;
		  
		  /**
		   * \brief The pair type used to synchronize vertex across across machines.
		   */
		  typedef std::pair<vertex_id_type, vertex_data_type> vid_vdata_pair_type;
		  
		  /**
		   * \brief The type of the exchange used to synchronize vertex data
		   */
		  typedef buffered_exchange<vid_vdata_pair_type> vdata_exchange_type;
		  
			  /**
			   * \brief The distributed exchange used to synchronize changes to
			   * vertex programs.
			   */
			  vdata_exchange_type vdata_exchange;
		  
			  /**
			   * \brief The pair type used to synchronize the results of the gather phase
			   */
			  typedef std::pair<vertex_id_type, gather_type> vid_gather_pair_type;
		  
			  /**
			   * \brief The type of the exchange used to synchronize gather
			   * accumulators
			   */
			  typedef buffered_exchange<vid_gather_pair_type> gather_exchange_type;
		  
			  /**
			   * \brief The distributed exchange used to synchronize gather
			   * accumulators.
			   */
			  gather_exchange_type gather_exchange;
		  
			  /**
			   * \brief The pair type used to synchronize messages
			   */
			  typedef std::pair<vertex_id_type, message_type> vid_message_pair_type;
		  
			  /**
			   * \brief The type of the exchange used to synchronize messages
			   */
			  typedef buffered_exchange<vid_message_pair_type> message_exchange_type;
		  
			  /**
			   * \brief The distributed exchange used to synchronize messages
			   */
			  message_exchange_type message_exchange;
		  
		  DECLARE_EVENT(EVENT_APPLIES);
		  DECLARE_EVENT(EVENT_GATHERS);
		  DECLARE_EVENT(EVENT_SCATTERS);
		  DECLARE_EVENT(EVENT_ACTIVE_CPUS);
	  
		public:
	  
		  /**
		   * Constructs an asynchronous consistent distributed engine.
		   * The number of threads to create are read from
		   * \ref graphlab_options::get_ncpus "opts.get_ncpus()". The scheduler to
		   * construct is read from
		   * \ref graphlab_options::get_scheduler_type() "opts.get_scheduler_type()".
		   * The default scheduler
		   * is the queued_fifo scheduler. For details on the scheduler types
		   * \see scheduler_types
		   *
		   *  See the <a href=#engineopts> main class documentation</a> for the
		   *  available engine options.
		   *
		   * \param dc Distributed controller to associate with
		   * \param graph The graph to schedule over. The graph must be fully
		   *			  constructed and finalized.
		   * \param opts A graphlab::graphlab_options object containing options and
		   *			 parameters for the scheduler and the engine.
		   */
		  xadaptive_engine(distributed_control& dc, graph_type& graph,
							 const graphlab_options& opts = graphlab_options());
		  
		  /* xie insert: have merged into constucture.	  
			  xxadaptive_engine(distributed_control &dc,
								  graph_type& graph,
								  const graphlab_options& opts = graphlab_options()) :
			  rmi(dc, this), graph(graph), scheduler_ptr(NULL),
			  aggregator(dc, graph, new context_type(*this, graph)), started(false),
			  engine_start_time(timer::approx_time_seconds()), force_stop(false) {
			rmi.barrier();
	  
			nfibers = 3000;
			stacksize = 16384;
			use_cache = false;
			factorized_consistency = true;
			timed_termination = (size_t)(-1);
			termination_reason = execution_status::UNSET;
			set_options(opts);
			initialize();
			rmi.barrier();
		  }*/
	  
		private:
	  
		  /**
		   * \internal
		   * Configures the engine with the provided options.
		   * The number of threads to create are read from
		   * opts::get_ncpus(). The scheduler to construct is read from
		   * graphlab_options::get_scheduler_type(). The default scheduler
		   * is the queued_fifo scheduler. For details on the scheduler types
		   * \see scheduler_types
		   */
	  
		  //xie insert: async engine set options
		  void xset_options(const graphlab_options& opts) {
			rmi.barrier();
			ncpus = opts.get_ncpus();
			ASSERT_GT(ncpus, 0);
			aggregation_lock.resize(opts.get_ncpus());
			aggregation_queue.resize(opts.get_ncpus());
			std::vector<std::string> keys = opts.get_engine_args().get_option_keys();
			
			foreach(std::string opt, keys) {
			  if (opt == "max_iterations") {
				  opts.get_engine_args().get_option("max_iterations", max_iterations);
				  //xie insert
				  has_max_iterations = true;
				  if (rmi.procid() == 0)
					logstream(LOG_EMPH) << "Engine Option: max_iterations = "
					  << max_iterations << std::endl;
			  } 
			  //mode setting 
			  else if(opt == "sample"){
				  bool smode = false; 
				  opts.get_engine_args().get_option("sample", smode);
				  if(smode){
					  running_mode = X_SAMPLE;
					  current_engine = X_ASYNC;
					  if (rmi.procid() == 0)
						logstream(LOG_EMPH) << "Engine Option: set mode X_SAMPLE"<< std::endl;
				  }
			  }
			  else if(opt == "auto"){
				  bool amode = false; 
				  opts.get_engine_args().get_option("auto", amode);
				  if(amode){
					running_mode = X_ADAPTIVE;
				  	if (rmi.procid() == 0)
						logstream(LOG_EMPH) << "Engine Option: set mode X_ADAPTIVE"<< std::endl;
				  }
			  }
			  else if(opt == "manual"){
				  bool rmode = false; 
				  opts.get_engine_args().get_option("manual", rmode);
				  if(rmode){
					running_mode = X_MANUAL;
					if (rmi.procid() == 0)
						logstream(LOG_EMPH) << "Engine Option: set mode X_MANUAL"<< std::endl;
				  }
			  }
			  //========manual setting
			  else if(opt == "s_iternum"){
				  opts.get_engine_args().get_option("s_iternum", switch_iter);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set SYNC run iters: "<< switch_iter << std::endl;
			  }
			  else if(opt == "a_tasknum"){
				  opts.get_engine_args().get_option("a_tasknum", tasknum);
				  tasknum = tasknum*10000;
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set ASYNC run tasknum: "<< tasknum << std::endl;
			  }
			  else if(opt == "a_thro"){
			  	  opts.get_engine_args().get_option("a_thro", thro_A);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set ASYNC throughput: "<< thro_A << std::endl;
			  	
			  }
			  else if(opt == "rate_AvsS"){
			  	  opts.get_engine_args().get_option("rate_AvsS", rate_AvsS);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set A/S rate: "<< rate_AvsS << std::endl;
			  	
			  }
			  //========= SYNC setting
			  else if(opt == "s_min_iterations"){
				  opts.get_engine_args().get_option("s_min_iterations", X_S_Min_Iters);
				  if (rmi.procid() == 0)
					logstream(LOG_EMPH) << "Engine Option: min_iterations = "
					  << X_S_Min_Iters << std::endl;
				  }
			  else if(opt == "s_increase_rate"){
				  opts.get_engine_args().get_option("s_increase_rate", X_S_Increase_Rate);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set SYNC increase_rate threshold "<< X_S_Increase_Rate << std::endl;
				  }
			  else if(opt == "s_sampled_iters"){
				  opts.get_engine_args().get_option("s_sampled_iters", X_S_Sampled_Iters);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set SYNC sampled_iters threshold "<< X_S_Sampled_Iters << std::endl;
				  }
			  //========= ASYNC setting
			  else if(opt=="t_sample"){
			  	  opts.get_engine_args().get_option("t_sample", T_SAMPLE);
				  if(rmi.procid() == 0)
					logstream(LOG_EMPH) << "Engine Option: t_sample "<<T_SAMPLE<< std::endl;
			  }
			  else if(opt == "start_async") {
				  bool start_a;
				  opts.get_engine_args().get_option("start_async", start_a);
				  if(start_a){
					  if(rmi.procid() == 0)
						logstream(LOG_EMPH) << "Engine Option: start with ASYNC mode "<<start_a<< std::endl;
					  current_engine = X_ASYNC;
				  }
				  else {
				  	if(rmi.procid() == 0)
						logstream(LOG_EMPH) << "Engine Option: start with SYNC mode "<<start_a<< std::endl;
				  		current_engine = X_SYNC;
				  }
			  }
			  else if(opt == "set_a_delay"){
				  //float threshold;
				  opts.get_engine_args().get_option("set_a_delay", X_A_Delay);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set a_delay threshold "<< X_A_Delay << std::endl;
				  }
			  else if(opt == "set_a_delay_counter"){
				  //float threshold;
				  opts.get_engine_args().get_option("set_a_delay_counter", X_A_Delay_Counter);
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set a_delay_counter threshold "<< X_A_Delay_Counter << std::endl;
				  }
			  else if(opt == "set_a_threshold"){
				  //float threshold;
				  opts.get_engine_args().get_option("set_a_threshold", X_A_Threshold_low);
				  X_A_Threshold_hig = X_A_Threshold_low*1.5;
		  
				  if (rmi.procid() == 0)
					  logstream(LOG_EMPH) << "Engine Option: set ASYNC threshold "<< X_A_Threshold_low << std::endl;
				  }
			  //xie insert end ===
			  else if (opt == "sched_allv") {
				  opts.get_engine_args().get_option("sched_allv", sched_allv);
				  if (rmi.procid() == 0)
					logstream(LOG_EMPH) << "Engine Option: sched_allv = "
					  << sched_allv << std::endl;
			  } else if (opt == "timeout") {
				opts.get_engine_args().get_option("timeout", timed_termination);
				if (rmi.procid() == 0)
				  logstream(LOG_EMPH) << "Engine Option: timeout = " << timed_termination << std::endl;
			  } else if (opt == "factorized") {
				opts.get_engine_args().get_option("factorized", factorized_consistency);
				if (rmi.procid() == 0)
				  logstream(LOG_EMPH) << "Engine Option: factorized = " << factorized_consistency << std::endl;
			  } else if (opt == "nfibers") {
				opts.get_engine_args().get_option("nfibers", nfibers);
				if (rmi.procid() == 0)
				  logstream(LOG_EMPH) << "Engine Option: nfibers = " << nfibers << std::endl;
			  } else if (opt == "stacksize") {
				opts.get_engine_args().get_option("stacksize", stacksize);
				if (rmi.procid() == 0)
				  logstream(LOG_EMPH) << "Engine Option: stacksize= " << stacksize << std::endl;
			  } else if (opt == "use_cache") {
				opts.get_engine_args().get_option("use_cache", use_cache);
				if (rmi.procid() == 0)
				  logstream(LOG_EMPH) << "Engine Option: use_cache = " << use_cache << std::endl;
			  } else {
				logstream(LOG_FATAL) << "Unexpected Engine Option: " << opt << std::endl;
			  }
			}
			opts_copy = opts;
			// set a default scheduler if none
			if (opts_copy.get_scheduler_type() == "") {
			  opts_copy.set_scheduler_type("queued_fifo");
			}
	  
			// construct scheduler passing in the copy of the options from set_options
			scheduler_ptr = scheduler_factory::
						  new_scheduler(graph.num_local_vertices(),
										opts_copy);
			rmi.barrier();
	  
			// create initial fork arrangement based on the alternate vid mapping
			if (factorized_consistency == false) {
			  cmlocks = new distributed_chandy_misra<graph_type>(rmi.dc(), graph,
														  boost::bind(&engine_type::xlock_ready, this, _1));
														  
			}
			else {
			  cmlocks = NULL;
			}
	  
			// construct the termination consensus object
			consensus = new fiber_async_consensus(rmi.dc(), nfibers);
		  }
	  
	  
		public:
		  ~xadaptive_engine() {
			delete consensus;
			delete cmlocks;
			delete scheduler_ptr;
		  }
	  
	  
	  
	  
		  // documentation inherited from iengine
		  /* xie insert merge two update counters
			  size_t xnum_updates() const {
				  return programs_executed.value;
			  }*/
	  
	  
	  
	  
	  
		  // documentation inherited from iengine
		  /* xie insert: have merged in elapsed_seconds(),	start_time is initialized in start();
			  float xelapsed_seconds() const {
				  return timer::approx_time_seconds() - engine_start_time;
			  }
		  */
	  
	  
		  /**
		   * \brief Not meaningful for the asynchronous engine. Returns -1.
		   */
		  /* xie insert:  async has no iteration.
			  int xiteration() const { return -1; }
		  */
	  
	  /**************************************************************************
	   *						   Signaling Interface							*
	   **************************************************************************/
	  
		private:
	  
		  /**
		   * \internal
		   * This is used to receive a message forwarded from another machine
		   */
		  void xrpc_signal(vertex_id_type vid,
								  const message_type& message) {
			if (force_stop) return;
			const lvid_type local_vid = graph.local_vid(vid);
			double priority;
			
			xmessages.add(local_vid, message, &priority);
			scheduler_ptr->schedule(local_vid, priority);
			consensus->cancel();
		  }
	  
		  /**
		   * \internal
		   * \brief Signals a vertex with an optional message
		   *
		   * Signals a vertex, and schedules it to be executed in the future.
		   * must be called on a vertex accessible by the current machine.
		   */
		  void xinternal_signal(const vertex_type& vtx,
							   const message_type& message = message_type()) {
			if (force_stop) return; 	
			if (started) {
			  const typename graph_type::vertex_record& rec = graph.l_get_vertex_record(vtx.local_id());
			  const procid_t owner = rec.owner;
			  if (endgame_mode) {
				// fast signal. push to the remote machine immediately
				if (owner != rmi.procid()) {
				  const vertex_id_type vid = rec.gvid;
				  rmi.remote_call(owner, &engine_type::xrpc_signal, vid, message);
				}
				else {
				  double priority;
				  xmessages.add(vtx.local_id(), message, &priority);
				  scheduler_ptr->schedule(vtx.local_id(), priority);
				  consensus->cancel();
				}
			  }
			  else {
				double priority;
				xmessages.add(vtx.local_id(), message, &priority);
				scheduler_ptr->schedule(vtx.local_id(), priority);
				consensus->cancel();
			  }
			}
			else {
			  double priority;
			  xmessages.add(vtx.local_id(), message, &priority);
			  scheduler_ptr->schedule(vtx.local_id(), priority);
			  consensus->cancel();
			}
		  } // end of schedule
	  
	  
		  /**
		   * \internal
		   * \brief Signals a vertex with an optional message
		   *
		   * Signals a global vid, and schedules it to be executed in the future.
		   * If current machine does not contain the vertex, it is ignored.
		   */
		  void xinternal_signal_gvid(vertex_id_type gvid,
									const message_type& message = message_type()) {
			if (force_stop) return;
			if (graph.is_master(gvid)) {
			  xinternal_signal(graph.vertex(gvid), message);
			}
		  } 
	  
	  
	  
		  void xinternal_signal_broadcast(vertex_id_type gvid,
										 const message_type& message = message_type()) {
			for (size_t i = 0;i < rmi.numprocs(); ++i) {
			  rmi.remote_call(i, &xadaptive_engine::xinternal_signal_gvid,
							  gvid, message);
			}
		  } // end of signal_broadcast
	  
	  
	  
		  void xrpc_internal_stop() {
			force_stop = true;
			termination_reason = execution_status::FORCED_ABORT;
		  }
	  
		  /**
		   * \brief Force engine to terminate immediately.
		   *
		   * This function is used to stop the engine execution by forcing
		   * immediate termination.
		   */
		  void xinternal_stop() {
			for (procid_t i = 0;i < rmi.numprocs(); ++i) {
			  rmi.remote_call(i, &xadaptive_engine::xrpc_internal_stop);
			}
		  }
	  
	  
	  
		  /**
		   * \brief Post a to a previous gather for a give vertex.
		   *
		   * This function is called by the \ref graphlab::context.
		   *
		   * @param [in] vertex The vertex to which to post a change in the sum
		   * @param [in] delta The change in that sum
		   */
		   
		  /* xie insert: have merged
			  void xinternal_post_delta(const vertex_type& vertex,
								   const gather_type& delta) {
			if(use_cache) {
			  const lvid_type lvid = vertex.local_id();
			  vertexlocks[lvid].lock();
			  if( has_cache.get(lvid) ) {
				gather_cache[lvid] += delta;
			  } else {
				// You cannot add a delta to an empty cache.  A complete
				// gather must have been run.
				// gather_cache[lvid] = delta;
				// has_cache.set_bit(lvid);
			  }
			  vertexlocks[lvid].unlock();
			}
		  }*/
	  
		  /**
		   * \brief Clear the cached gather for a vertex if one is
		   * available.
		   *
		   * This function is called by the \ref graphlab::context.
		   *
		   * @param [in] vertex the vertex for which to clear the cache
		   */
		  /* xie insert: have been merged.
		  void xinternal_clear_gather_cache(const vertex_type& vertex) {
			const lvid_type lvid = vertex.local_id();
			if(use_cache && has_cache.get(lvid)) {
			  vertexlocks[lvid].lock();
			  gather_cache[lvid] = gather_type();
			  has_cache.clear_bit(lvid);
			  vertexlocks[lvid].unlock();
			}
	  
		  }*/
	  
		public:
	  
	  
	  
		  /*void xsignal(vertex_id_type gvid,
					  const message_type& message = message_type()) {
			rmi.barrier();
			xinternal_signal_gvid(gvid, message);
			rmi.barrier();
		  }*/
	  
	  
		  void xsignal_all(const message_type& message = message_type(),
						  const std::string& order = "shuffle") {
			vertex_set vset = graph.complete_set();
			signal_vset(vset, message, order);
		  } // end of schedule all
	  
		  void xsignal_vset(const vertex_set& vset,
						  const message_type& message = message_type(),
						  const std::string& order = "shuffle") {
			logstream(LOG_DEBUG) << rmi.procid() << ": Schedule All" << std::endl;
			// allocate a vector with all the local owned vertices
			// and schedule all of them.
			std::vector<vertex_id_type> vtxs;
			vtxs.reserve(graph.num_local_own_vertices());
			for(lvid_type lvid = 0;
				lvid < graph.get_local_graph().num_vertices();
				++lvid) {
			  if (graph.l_vertex(lvid).owner() == rmi.procid() &&
				  vset.l_contains(lvid)) {
				vtxs.push_back(lvid);
			  }
			}
	  
			if(order == "shuffle") {
			  graphlab::random::shuffle(vtxs.begin(), vtxs.end());
			}
			foreach(lvid_type lvid, vtxs) {
			  double priority;
			  xmessages.add(lvid, message, &priority);
			  scheduler_ptr->schedule(lvid, priority);
			}
			rmi.barrier();
		  }
	  
		  void x_inner_signal_vset(const std::string& order = "shuffle") {			  
			//logstream(LOG_DEBUG) << rmi.procid() << ": Schedule ~~~~~~~~" << std::endl;
	  
			// allocate a vector with all the local owned vertices
			// and schedule all of them.
			std::vector<vertex_id_type> vtxs;
			vtxs.reserve(graph.num_local_own_vertices());
			for(lvid_type lvid = 0;
				lvid < graph.get_local_graph().num_vertices();
				++lvid) {
			  if (graph.l_vertex(lvid).owner() == rmi.procid() &&
				  next_mode_active_vertex.get(lvid)) {
				vtxs.push_back(lvid);
			  }
			}
	  
			if(order == "shuffle") {
			  graphlab::random::shuffle(vtxs.begin(), vtxs.end());
			}
			
			foreach(lvid_type lvid, vtxs) {
			  //logstream(LOG_INFO) << rmi.procid() << ": Schedule ~~~~~~~~ "<<lvid << std::endl;
			  double priority;
			  xmessages.add(lvid, messages[lvid], &priority);
			  // xie insert : clear messages
			  messages[lvid] = message_type();
			  
			  scheduler_ptr->schedule(lvid, priority);
			}
			rmi.barrier();
		  }
	  
		private: 
	  
		  /**
		   * Gets a task from the scheduler and the associated message
		   */
		  sched_status::status_enum xget_next_sched_task( size_t threadid,
														lvid_type& lvid,
														message_type& msg) {
			while (1) {
			  sched_status::status_enum stat = 
				  scheduler_ptr->get_next(threadid % ncpus, lvid);
			  if (stat == sched_status::NEW_TASK) {
				if (xmessages.get(lvid, msg)) return stat;
				else continue;
			  }
			  return stat;
			}
		  }
	  
		  void xset_endgame_mode() {
			  //if (!endgame_mode) logstream(LOG_EMPH) << "Endgame mode\n";
			  endgame_mode = true;
			  rmi.dc().set_fast_track_requests(true);
		  } 
	  
		  
		  void xset_stop_async() {
				 stop_async = true;
				 consensus->force_done();
				 rmi.dc().set_fast_track_requests(true);
			 } 
	  
		  /**
		   * \internal
		   * Called when get_a_task returns no internal task not a scheduler task.
		   * This rechecks the status of the internal task queue and the scheduler
		   * inside a consensus critical section.
		   */
		  bool xtry_to_quit(size_t threadid,
						   bool& has_sched_msg,
						   lvid_type& sched_lvid,
						   message_type &msg) {
			if (timer::approx_time_seconds() - engine_start_time > timed_termination) {
			  termination_reason = execution_status::TIMEOUT;
			  force_stop = true;
			}
			fiber_control::yield();
			logstream(LOG_DEBUG) << rmi.procid() << "-" << threadid << ": " << "Termination Attempt " << std::endl;
			has_sched_msg = false;
	  
			//xie insert
			if(stop_async) 
			  return true;
			//logstream(LOG_INFO) << rmi.procid() << "-" << threadid << ": " << "cratical before " << std::endl;
	  
			
			consensus->begin_done_critical_section(threadid);
			sched_status::status_enum stat = 
				xget_next_sched_task(threadid, sched_lvid, msg);
	  
			
			if (stat == sched_status::EMPTY || force_stop) {
			  //logstream(LOG_INFO) << rmi.procid() << "-" << threadid <<  ": "
			  //					   << "\tTermination Double Checked" 
			  //					   << std::endl;
	  
			  if(!endgame_mode){
			  	  logstream(LOG_EMPH) << "Endgame mode\n";
				  endgame_mode = true;
				  // put everyone in endgame
				  for (procid_t i = 0;i < rmi.dc().numprocs(); ++i) {
					rmi.remote_call(i, &xadaptive_engine::xset_endgame_mode);
				  } 
			  }
			  
			  bool ret = consensus->end_done_critical_section(threadid);
	  
			  //xie insert
			  //logstream(LOG_INFO) << rmi.procid() << "-" << threadid << ": " << "cratical end " << std::endl;
			  if(stop_async) 
				  return true;
			  
			  if (ret == false) {
				logstream(LOG_DEBUG) << rmi.procid() << "-" << threadid <<	": "
								   << "\tCancelled"<< std::endl;
			  } else {
				logstream(LOG_DEBUG) << rmi.procid() << "-" << threadid <<	": "
								   << "\tDying" << " (" << fiber_control::get_tid() << ")" << std::endl;
			  }
			  return ret;
			} 
			//xie insert: if has task, and stop_async=true, just deal with it and then stop
			else {
			  logstream(LOG_DEBUG) << rmi.procid() << "-" << threadid <<  ": "
								   << "\tCancelled by Scheduler Task" << std::endl;
			  consensus->cancel_critical_section(threadid);
			  has_sched_msg = true;
			  return false;
			}
		  } // end of try to quit
	  
	  
		  /**
		   * \internal
		   * When all distributed locks are acquired, this function is called
		   * from the chandy misra implementation on the master vertex.
		   * Here, we perform initialization
		   * of the task and switch the vertex to a gathering state
		   */
		  void xlock_ready(lvid_type lvid) {
			cm_handles[lvid]->lock.lock();
			cm_handles[lvid]->philosopher_ready = true;
			fiber_control::schedule_tid(cm_handles[lvid]->fiber_handle);
			cm_handles[lvid]->lock.unlock();
		  }
	  
	  
		  conditional_gather_type xperform_gather(vertex_id_type vid,
									 vertex_program_type& vprog_) {
			vertex_program_type vprog = vprog_;
			lvid_type lvid = graph.local_vid(vid);
			local_vertex_type local_vertex(graph.l_vertex(lvid));
			vertex_type vertex(local_vertex);
			context_type context(*this, graph);
			edge_dir_type gather_dir = vprog.gather_edges(context, vertex);
			conditional_gather_type accum;
	  
			
			//check against the cache
			if( use_cache && has_cache.get(lvid) ) {
				accum.set(gather_cache[lvid]);
				return accum;
			}
			// do in edges
			if(gather_dir == IN_EDGES || gather_dir == ALL_EDGES) {
			  foreach(local_edge_type local_edge, local_vertex.in_edges()) {
				edge_type edge(local_edge);
				lvid_type a = edge.source().local_id(), b = edge.target().local_id();
				vertexlocks[std::min(a,b)].lock();
				vertexlocks[std::max(a,b)].lock();
				accum += vprog.gather(context, vertex, edge);
				vertexlocks[a].unlock();
				vertexlocks[b].unlock();
			  }
			} 
			// do out edges
			if(gather_dir == OUT_EDGES || gather_dir == ALL_EDGES) {
			  foreach(local_edge_type local_edge, local_vertex.out_edges()) {
				edge_type edge(local_edge);
				lvid_type a = edge.source().local_id(), b = edge.target().local_id();
				vertexlocks[std::min(a,b)].lock();
				vertexlocks[std::max(a,b)].lock();
				accum += vprog.gather(context, vertex, edge);
				vertexlocks[a].unlock();
				vertexlocks[b].unlock();
			  }
			} 
			if (use_cache) {
			  gather_cache[lvid] = accum.value; has_cache.set_bit(lvid);
			}
			return accum;
		  }
	  
	  
		  void xperform_scatter_local(lvid_type lvid,
									 vertex_program_type& vprog) {
			local_vertex_type local_vertex(graph.l_vertex(lvid));
			vertex_type vertex(local_vertex);
			context_type context(*this, graph);
			edge_dir_type scatter_dir = vprog.scatter_edges(context, vertex);
			if(scatter_dir == IN_EDGES || scatter_dir == ALL_EDGES) {
			  foreach(local_edge_type local_edge, local_vertex.in_edges()) {
				edge_type edge(local_edge);
				lvid_type a = edge.source().local_id(), b = edge.target().local_id();
				vertexlocks[std::min(a,b)].lock();
				vertexlocks[std::max(a,b)].lock();
				vprog.scatter(context, vertex, edge);
				vertexlocks[a].unlock();
				vertexlocks[b].unlock();
			  }
			} 
			if(scatter_dir == OUT_EDGES || scatter_dir == ALL_EDGES) {
			  foreach(local_edge_type local_edge, local_vertex.out_edges()) {
				edge_type edge(local_edge);
				lvid_type a = edge.source().local_id(), b = edge.target().local_id();
				vertexlocks[std::min(a,b)].lock();
				vertexlocks[std::max(a,b)].lock();
				vprog.scatter(context, vertex, edge);
				vertexlocks[a].unlock();
				vertexlocks[b].unlock();
			  }
			} 
	  
			// release locks
			if (!factorized_consistency) {
			  cmlocks->philosopher_stops_eating_per_replica(lvid);
			}
		  }
	  
	  
		  void xperform_scatter(vertex_id_type vid,
						  vertex_program_type& vprog_,
						  const vertex_data_type& newdata) {
			vertex_program_type vprog = vprog_;
			lvid_type lvid = graph.local_vid(vid);
			vertexlocks[lvid].lock();
			graph.l_vertex(lvid).data() = newdata;
			vertexlocks[lvid].unlock();
			xperform_scatter_local(lvid, vprog);
		  }
	  
		  //xie insert
		  void xperform_vdata_sync(vertex_id_type vid,
							  const vertex_data_type& newdata) {
				lvid_type lvid = graph.local_vid(vid);
				vertexlocks[lvid].lock();
				graph.l_vertex(lvid).data() = newdata;
				vertexlocks[lvid].unlock();
				 // release locks
				if (!factorized_consistency) {
				  cmlocks->philosopher_stops_eating_per_replica(lvid);
				}
		  }
	  
	  
		  // make sure I am the only person running.
		  // if returns false, the message has been dropped into the message array.
		  // quit
		  bool xget_exclusive_access_to_vertex(const lvid_type lvid,
											  const message_type& msg) {
			vertexlocks[lvid].lock();
			bool someone_else_running = program_running.set_bit(lvid);
			if (someone_else_running) {
			  // bad. someone else is here.
			  // drop it into the message array 	  
			  xmessages.add(lvid, msg);
			  hasnext.set_bit(lvid);
			  exclusive_executed.inc();
			} 
			vertexlocks[lvid].unlock();
			return !someone_else_running;
		  }
	  
	  
	  
		  // make sure I am the only person running.
		  // if returns false, the message has been dropped into the message array.
		  // quit
		  void xrelease_exclusive_access_to_vertex(const lvid_type lvid) {
			vertexlocks[lvid].lock();
			// someone left a next message for me
			// reschedule it at high priority
			if (hasnext.get(lvid)) {
			  scheduler_ptr->schedule(lvid, 10000.0);
			  consensus->cancel();
			  hasnext.clear_bit(lvid);
			}
			program_running.clear_bit(lvid);
			vertexlocks[lvid].unlock();
		  }
	  
	  
		  /**
		   * \internal
		   * Called when the scheduler returns a vertex to run.
		   * If this function is called with vertex locks acquired, prelocked
		   * should be true. Otherwise it should be false.
		   */
		  void xeval_sched_task(const lvid_type lvid,
							   const message_type& msg, size_t threadid) {
			const typename graph_type::vertex_record& rec = graph.l_get_vertex_record(lvid);
			vertex_id_type vid = rec.gvid;
			// if this is another machine's forward it
			if (rec.owner != rmi.procid()) {
			  rmi.remote_call(rec.owner, &engine_type::xrpc_signal, vid, msg);
			  return;
			}
			// I have to run this myself
			
			if (!xget_exclusive_access_to_vertex(lvid, msg)) return;
	  
			//startcounter[fiber_control::get_worker_id()]++;
			
			/**************************************************************************/
			/*							   Acquire Locks							  */
			/**************************************************************************/
			if (!factorized_consistency) {
			  // begin lock acquisition
			  cm_handles[lvid] = new vertex_fiber_cm_handle;
			  cm_handles[lvid]->philosopher_ready = false;
			  cm_handles[lvid]->fiber_handle = fiber_control::get_tid();
			  cmlocks->make_philosopher_hungry(lvid);
			  cm_handles[lvid]->lock.lock();
			  while (!cm_handles[lvid]->philosopher_ready) {
				fiber_control::deschedule_self(&(cm_handles[lvid]->lock.m_mut));
				cm_handles[lvid]->lock.lock();
			  }
			  cm_handles[lvid]->lock.unlock();
			}
			/**************************************************************************/
			/*							   Begin Program							  */
			/**************************************************************************/
			context_type context(*this, graph);
			vertex_program_type vprog = vertex_program_type();
			local_vertex_type local_vertex(graph.l_vertex(lvid));
			vertex_type vertex(local_vertex);
	  
			/**************************************************************************/
			/*								 init phase 							  */
			/**************************************************************************/
			vprog.init(context, vertex, msg);
	  
			/**************************************************************************/
			/*								Gather Phase							  */
			/**************************************************************************/
			
			conditional_gather_type gather_result;
			std::vector<request_future<conditional_gather_type> > gather_futures;
	  
			if(vprog.gather_edges(context, vertex)!=graphlab::NO_EDGES)  //xie insert change:  judge if no_edges previously
			{	 foreach(procid_t mirror, local_vertex.mirrors()) {
				  gather_futures.push_back(
					  object_fiber_remote_request(rmi, 
												  mirror, 
												  &xadaptive_engine::xperform_gather, 
												  vid,
												  vprog));
				}
				gather_result += xperform_gather(vid, vprog);
						
				for(size_t i = 0;i < gather_futures.size(); ++i) {
				  gather_result += gather_futures[i]();
				}
			}// end skip no_edges
	  
		   /**************************************************************************/
		   /*							   apply phase								 */
		   /**************************************************************************/ 
	  
		   //xie insert tmp :
		   //const vertex_data_type predata = local_vertex.data();
		   
		   vertexlocks[lvid].lock();
		   vprog.apply(context, vertex, gather_result.value);	   
		   vertexlocks[lvid].unlock();
	  
	  
		   /**************************************************************************/
		   /*							 scatter phase								 */
		   /**************************************************************************/
	  
		   // should I wait for the scatter? nah... but in case you want to
		   // the code is commented below
		   /*foreach(procid_t mirror, local_vertex.mirrors()) {
			  rmi.remote_call(mirror, 
							 &xadaptive_engine::perform_scatter, 
							 vid,
							 vprog,
							 local_vertex.data());
				  }*/
	  
		   
		   /*if(vprog.scatter_edges(context, vertex)==graphlab::NO_EDGES){		  //xie insert skip no scatter
			std::vector<request_future<void> > scatter_futures;
			foreach(procid_t mirror, local_vertex.mirrors()) {
				 scatter_futures.push_back(
					 object_fiber_remote_request(rmi, 
												 mirror, 
												 &xadaptive_engine::xperform_vdata_sync, 
												 vid,
												 local_vertex.data()));
			   }
	  
			 for(size_t i = 0;i < scatter_futures.size(); ++i) 
				 scatter_futures[i]();
		   }
		   // xie insert end 
		   else */				
	  
		   {
			   std::vector<request_future<void> > scatter_futures;
			   foreach(procid_t mirror, local_vertex.mirrors()) {
				 scatter_futures.push_back(
					 object_fiber_remote_request(rmi, 
												 mirror, 
												 &xadaptive_engine::xperform_scatter, 
												 vid,
												 vprog,
												 local_vertex.data()));
			   }
			   xperform_scatter_local(lvid, vprog);
			   for(size_t i = 0;i < scatter_futures.size(); ++i) 
				 scatter_futures[i]();
		   }
		   
		   
			/************************************************************************/
			/*							 Release Locks								*/
			/************************************************************************/
			// the scatter is used to release the chandy misra
			// here I cleanup
			if (!factorized_consistency) {
			  delete cm_handles[lvid];
			  cm_handles[lvid] = NULL;
			}
			xrelease_exclusive_access_to_vertex(lvid);
			programs_executed.inc(); 
	  
			//if(fiber_control::get_worker_id()==0) 
			{
				  //if((rmi.procid()==0)&&(endcounter[0]%500==0)/*||(fiber_control::get_worker_id()==21)*/)
				  /*{
						  size_t scounter = 0;
						  size_t ecounter = 0;
						for(size_t i=0;i<30; i++){
							scounter+=startcounter[i];
							ecounter+=endcounter[i];
							}
						double now = globaltimer.current_time_millis();
						logstream(LOG_INFO)<<rmi.procid()<<"-"<<fiber_control::get_worker_id()
							<<" s&e "<<scounter<<" "<<ecounter
							<<" differ "<<(scounter-ecounter)
							<<" s&e_a "<<(scounter-lastcounters)/(now-lastime)
							<<" "<<(ecounter-lastcounter)/(now-lastime)
							<<" At "<<now<<std::endl; 
	  
					   lastcounter = ecounter;
					   lastcounters = scounter;
					   lastime = globaltimer.current_time_millis();
					  }
	  */
				  /*if((rmi.procid()==0)&&(a_numcounter%50==0))
				  {
					  logstream(LOG_INFO)<<rmi.procid()<<"-"<<fiber_control::get_worker_id()
						  <<": G "<<a_gcounter/a_numcounter
						  <<",A "<<a_acounter/a_numcounter
						  <<",S "<<a_scounter/a_numcounter
						  <<",Other "<<a_othercounter/a_numcounter<<std::endl;
					  
					  a_gcounter = 0;
					  a_acounter = 0;
					  a_scounter = 0;
					  a_othercounter = 0;
					  a_numcounter = 0;
				  }*/
		  }
		  //endcounter[fiber_control::get_worker_id()]++;
		  }
	  
	  
		  /**
		   * \internal
		   * Per thread main loop
		   */
		  void xthread_start(size_t threadid) {
			bool has_sched_msg = false;
			std::vector<std::vector<lvid_type> > internal_lvid;
			lvid_type sched_lvid;
	  
			message_type msg;
			float last_aggregator_check = timer::approx_time_seconds();
			timer ti; ti.start();
	  
		  
			if((threadid%3001==0)){
				iteration_counter = 0;
				for(int i=0; i<11;i++){
				  avg_line[i] = 0;
				  active[i] = 0;
				  }
			}
			double xstart = globaltimer.current_time_millis();	   
		    double lastsampled = xstart;
			double lastexecuted = 0;
			size_t lastadd = 0;
			size_t count = 0;
			
			while(1) {
			  if(stop_async)  break;
			  
			  if (timer::approx_time_seconds() != last_aggregator_check && !endgame_mode) {
				last_aggregator_check = timer::approx_time_seconds();
				std::string key = aggregator.tick_asynchronous();
				if (key != "") {
				  for (size_t i = 0;i < aggregation_lock.size(); ++i) {
					aggregation_lock[i].lock();
					aggregation_queue[i].push_back(key);
					aggregation_lock[i].unlock();
				  }
				}
			  }
	  
	  
			  // test the aggregator
			  while(!aggregation_queue[fiber_control::get_worker_id()].empty()) {
				size_t wid = fiber_control::get_worker_id();
				ASSERT_LT(wid, ncpus);
				aggregation_lock[wid].lock();
				std::string key = aggregation_queue[wid].front();
				aggregation_queue[wid].pop_front();
				aggregation_lock[wid].unlock();
				aggregator.tick_asynchronous_compute(wid, key);
			  }
	  
	  
			  sched_status::status_enum stat = xget_next_sched_task(threadid, sched_lvid, msg);
	  
	  
			  has_sched_msg = stat != sched_status::EMPTY;
			  if (stat != sched_status::EMPTY) {
				xeval_sched_task(sched_lvid, msg, threadid);
				if (endgame_mode) rmi.dc().flush();
			  }
			  else if (!xtry_to_quit(threadid, has_sched_msg, sched_lvid, msg)) {
				/*
				* We failed to obtain a task, try to quit
				*/
				if (has_sched_msg) { 
					  xeval_sched_task(sched_lvid, msg,threadid);
				}
			  } else { 
				  break; 
			  }
	  
			  //xie insert
			  if(threadid%3001==0){
			  	if(running_mode==X_SAMPLE){
				  	double nowtime = globaltimer.current_time_millis();
					if(programs_executed.value>tasknum)
					{
						throughput = (programs_executed.value-lastexecuted)/(nowtime-lastsampled);
						stop_async = true;
						if(rmi.procid()==0)
						  	logstream(LOG_INFO)<< 0 << ": -------thro_a--------- "<<throughput<<std::endl;
						lastexecuted = programs_executed.value;
						lastsampled = globaltimer.current_time_millis();
					}
				}
				else {
				  double avg_inc_rate = 0;
				  double durtime = globaltimer.current_time_millis()-lastsampled;
				  if(durtime>T_SAMPLE){
					  size_t tmpact = xmessages.num_act();
					  size_t tmpexec = programs_executed.value;
	
					  size_t now = iteration_counter%11; 
					  if(tmpact>=tmpexec)
					  	active[now] = tmpact-tmpexec;
					  else active[now]=0;
					  
					  {
						  avg_line[now] = avg_line[(iteration_counter-1)%11]-(active[(iteration_counter-A_Sampled_Iters+11)%11]-active[now])/A_Sampled_Iters;
						  avg_inc_rate = (avg_line[now]-avg_line[(iteration_counter+6)%11])/avg_line[(iteration_counter+6)%11];
					  }

					  if(running_mode==X_ADAPTIVE){
						  double comparable = thro_A*durtime/rate_AvsS;
						  if(rmi.procid()==0)
							  logstream(LOG_EMPH)<< rmi.procid() << ": ------- sample ---"<<iteration_counter<<"--- "
							  		<<avg_inc_rate
									//<<" ,actn "<<active[now]
									//<<" ,tmpact "<<tmpact
									//<<" ,thro_A*durtime/rate_AvsS "<<comparable
									//<<std::endl
						  			<<" lastadd "<<lastadd
						  			<<" now "<<active[now]
						  			<<" now_avg "<<avg_line[now]
									<<" executed "<<tmpexec-lastexecuted
									<<std::endl;
						  					
						  if((avg_inc_rate>0)&&(lastadd>(thro_A*durtime)))
						  {
						  	  count++;
							  if(count>1){
						  	  first_time_start = false;
							  //set prepare to stop
							  stop_async = true;
							  //if(rmi.procid()==0)
							  logstream(LOG_EMPH)<< rmi.procid() << ": -------start switch ---"<<iteration_counter<<"--- "
							  		<<avg_inc_rate
									<<" ,lastadd "<<lastadd
									<<" ,executed "<<tmpexec-lastexecuted
									<<std::endl;
							  countoverhead = globaltimer.current_time_millis();
							  // put everyone in switch mode
							  for (procid_t i = 0;i < rmi.dc().numprocs(); ++i)
							 		  rmi.remote_call(i, &xadaptive_engine::xset_stop_async);

							  }
						  }
						  else count = 0;
						  
						  lastexecuted = tmpexec;
						  lastadd = active[now];
					  }
					  
					  lastsampled = globaltimer.current_time_millis();
					  ++iteration_counter;
				  }

				  if(running_mode==X_MANUAL){
					  if((avg_inc_rate>0)&&(programs_executed.value>tasknum)){
						  first_time_start = false;
						  //set prepare to stop
						  stop_async = true;
						  logstream(LOG_EMPH)<< rmi.procid() << ": -------start switch ------ "<<avg_inc_rate<<", task "<<programs_executed.value<<std::endl;
						  countoverhead = globaltimer.current_time_millis();
						  // put everyone in endgame
						  for (procid_t i = 0;i < rmi.dc().numprocs(); ++i)
								  rmi.remote_call(i, &xadaptive_engine::xset_stop_async);
					  }
				  }
				  
				}
					  
			  }
	
			
			  if(stop_async)  break;
	  
			  if (fiber_control::worker_has_fibers_on_queue()) fiber_control::yield();
			}
		  } // end of thread start
	  
	  
			  
	  /**************************************************************************
	   *						 Main engine start()							*
	   **************************************************************************/
	  
		public:
	  
	  
		  /**
			* \brief Start the engine execution.
			*
			* This function starts the engine and does not
			* return until the scheduler has no tasks remaining.
			*
			* \return the reason for termination
			*/
		  execution_status::status_enum xstart() {
			//xie insert
			current_engine = X_ASYNC;
			local_sync = false;   //whether start local sync  --haven't implemented
			stop_async = false;   //whether stop to switch mode
			d_join = 0;
			d_add = 0;
			d_complete = 0;
			delay = 0;
			graphlab::timer timer; timer.start();
	  
			
			//signal active vertex:  first signal then clear   the "next_mode_active_vertex" is just a pointer, point to asy_start_active_v or active_superstep
			if(!first_time_start) 
			  x_inner_signal_vset();
			asy_start_active_v.clear();
	  
			
			bool old_fasttrack = rmi.dc().set_fast_track_requests(false);
			if (rmi.procid() == 0)
			  logstream(LOG_INFO) << "Spawning " << nfibers << " threads" << std::endl;
			ASSERT_TRUE(scheduler_ptr != NULL);
			consensus->reset();
	  
			// now. It is of critical importance that we match the number of 
			// actual workers
		   
	  
			// start the aggregator
			aggregator.start(ncpus);
			aggregator.aggregate_all_periodic();
	  
			started = true;
	  
			rmi.barrier();
			size_t allocatedmem = memory_info::allocated_bytes();
			rmi.all_reduce(allocatedmem);
	  
			engine_start_time = timer::approx_time_seconds();
			force_stop = false;
			endgame_mode = true;//false;	  
			programs_executed = 0;
			exclusive_executed = 0; 	  //xie insert
			exclusive_executed_pre = 0;   //xie insert
			launch_timer.start();
	  
			termination_reason = execution_status::RUNNING;
			if (rmi.procid() == 0) {
			  logstream(LOG_INFO) << "Total Allocated Bytes: " << allocatedmem << std::endl;
			}
			fiber_group::affinity_type affinity;
			affinity.clear();
			for (size_t i = 0; i < ncpus; ++i) {
			  affinity.set_bit(i);
			}
			thrgroup.set_affinity(affinity);
			thrgroup.set_stacksize(stacksize);
	  
			//xie insert:  samppling
			//startcounter = 0;
			//endcounter = 0;
			lastcounter = 0;
			lastcounters = 0;
			sample_start = globaltimer.current_time_millis();
			double xstartime = lastime;
			
			for(size_t i = 0; i<30; ++i){
			  startcounter[i] = 0;
			  endcounter[i] = 0;
			  }
	  
			//xie insert record overhead
			if((rmi.procid()==0)&&(!first_time_start))
			  logstream(LOG_INFO)<<rmi.procid()<<"-"<<fiber_control::get_worker_id()
				  <<"  SYNC-ASYNC switch overhead "<<globaltimer.current_time_millis()-countoverhead<<std::endl;
			
			for (size_t i = 0; i < nfibers ; ++i) {
			  thrgroup.launch(boost::bind(&engine_type::xthread_start, this, i));
			}
			thrgroup.join();
			aggregator.stop();
	  
			
			  size_t scounter = 0;
			  size_t ecounter = 0;
			for(size_t i=0;i<30; i++){
				scounter+=startcounter[i];
				ecounter+=endcounter[i];
				}				
	  
			//if need to activate next mode
			if(stop_async){
			  rmi.full_barrier();
			  if(running_mode==X_SAMPLE){
			  	termination_reason = execution_status::TASK_DEPLETION;

			  double local_thro = throughput;
			  rmi.all_reduce(local_thro);
			  rmi.cout()<<"================ sampled result ============="<<std::endl
			  		<<"throughput "<<(local_thro/rmi.numprocs())
					<<" #e/#n "<<(graph.num_edges()*1.0/graph.num_vertices())
					<<" r "<<(graph.num_replicas()*1.0/graph.num_vertices())
					<<std::endl;
			  }
			  else{
				  //if(rmi.procid()==0)
				  logstream(LOG_INFO)<< "from async to sync now: "<<stop_async <<std::endl;
				  
				  //messages should be sent into next mode then clear
				  next_mode_active_vertex = xmessages.active_v;
				  termination_reason = execution_status::MODE_SWITCH;
			  }
			}
		  //xie insert: end of local switch range
		  
			// if termination reason was not changed, then it must be depletion
			if (termination_reason == execution_status::RUNNING) {
			  termination_reason = execution_status::TASK_DEPLETION;
			}
	  
			size_t ctasks = programs_executed.value;
			rmi.all_reduce(ctasks);
			programs_executed.value = ctasks;
	  
			rmi.cout() << "Completed Tasks: " << programs_executed.value << std::endl;
	  
	  
			size_t numjoins = xmessages.num_joins();
			rmi.all_reduce(numjoins);
			rmi.cout() << "Schedule Joins: " << numjoins << std::endl;
	  
			size_t numadds = xmessages.num_adds();
			rmi.all_reduce(numadds);
			rmi.cout() << "Schedule Adds: " << numadds <<" , End at "<<globaltimer.current_time_millis()<< std::endl;
	  
	  
			if(!stop_async)   //xie insert judgement
			  ASSERT_TRUE(scheduler_ptr->empty());
			started = false;
	  
			rmi.dc().set_fast_track_requests(old_fasttrack);
			return termination_reason;
		  } // end of xstart
	  
	  
		public:
		  //xie insert: same as in sync engine.   aggregator_type* xget_aggregator() { return &aggregator; }
	  
	  
		  /*
		  *   ======================================================================
		  *   ========================= start sync engine part code ========================
		  */
	  
		  /**
		   * \brief Start execution of the synchronous engine.
		   *
		   * The start function begins computation and does not return until
		   * there are no remaining messages or until max_iterations has
		   * been reached.
		   *
		   * The start() function modifies the data graph through the vertex
		   * programs and so upon return the data graph should contain the
		   * result of the computation.
		   *
		   * @return The reason for termination
		   */
		  execution_status::status_enum sstart();
		  execution_status::status_enum start();
	  
		  // documentation inherited from iengine
		  size_t num_updates() const;
	  
		  // documentation inherited from iengine
		  void signal(vertex_id_type vid,
					  const message_type& message = message_type());
	  
		  // documentation inherited from iengine
		  void signal_all(const message_type& message = message_type(),
						  const std::string& order = "shuffle");
	  
		  void signal_vset(const vertex_set& vset,
						  const message_type& message = message_type(),
						  const std::string& order = "shuffle");
		  
		  //xie insert: signal when form ASYCN ->SYNC
		  void s_inner_signal_vset( const std::string& order = "shuffle" );
	  
		  // documentation inherited from iengine
		  float elapsed_seconds() const;
	  
		  /**
		   * \brief Get the current iteration number since start was last
		   * invoked.
		   *
		   *  \return the current iteration
		   */
		  int iteration() const;
	  
	  
		  /**
		   * \brief Compute the total memory used by the entire distributed
		   * system.
		   *
		   * @return The total memory used in bytes.
		   */
		  size_t total_memory_usage() const;
	  
		  /**
		   * \brief Get a pointer to the distributed aggregator object.
		   *
		   * This is currently used by the \ref graphlab::iengine interface to
		   * implement the calls to aggregation.
		   *
		   * @return a pointer to the local aggregator.
		   */
		  aggregator_type* get_aggregator();
	  
		  /**
		   * \brief Initialize the engine and allocate datastructures for vertex, and lock,
		   * clear all the messages.
		   */
		  void init();
	  
	  
		private:
	  
	  
		  /**
		   * \brief Resize the datastructures to fit the graph size (in case of dynamic graph). Keep all the messages
		   * and caches.
		   */
		  void resize();
	  
		  /**
		   * \brief This internal stop function is called by the \ref graphlab::context to
		   * terminate execution of the engine.
		   */
		  void internal_stop();
	  
		  /**
		   * \brief This function is called remote by the rpc to force the
		   * engine to stop.
		   */
		  void rpc_stop();
	  
		  /**
		   * \brief Signal a vertex.
		   *
		   * This function is called by the \ref graphlab::context.
		   *
		   * @param [in] vertex the vertex to signal
		   * @param [in] message the message to send to that vertex.
		   */
		  void internal_signal(const vertex_type& vertex,
							   const message_type& message = message_type());
	  
		  /**
		   * \brief Called by the context to signal an arbitrary vertex.
		   * This must be done by finding the owner of that vertex.
		   *
		   * @param [in] gvid the global vertex id of the vertex to signal
		   * @param [in] message the message to send to that vertex.
		   */
		  void internal_signal_broadcast(vertex_id_type gvid,
										 const message_type& message = message_type());
	  
		  /**
		   * \brief This function tests if this machine is the master of
		   * gvid and signals if successful.
		   */
		  void internal_signal_rpc(vertex_id_type gvid,
									const message_type& message = message_type());
	  
	  
		  /**
		   * \brief Post a to a previous gather for a give vertex.
		   *
		   * This function is called by the \ref graphlab::context.
		   *
		   * @param [in] vertex The vertex to which to post a change in the sum
		   * @param [in] delta The change in that sum
		   */
		  void internal_post_delta(const vertex_type& vertex,
								   const gather_type& delta);
	  
		  /**
		   * \brief Clear the cached gather for a vertex if one is
		   * available.
		   *
		   * This function is called by the \ref graphlab::context.
		   *
		   * @param [in] vertex the vertex for which to clear the cache
		   */
		  void internal_clear_gather_cache(const vertex_type& vertex);
	  
	  
		  // Program Steps ==========================================================
	  
	  
		  void thread_launch_wrapped_event_counter(boost::function<void(void)> fn) {
			INCREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
			fn();
			DECREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
		  }
	  
		  /**
		   * \brief Executes ncpus copies of a member function each with a
		   * unique consecutive id (thread id).
		   *
		   * This function is used by the main loop to execute each of the
		   * stages in parallel.
		   *
		   * The member function must have the type:
		   *
		   * \code
		   * void synchronous_engine::member_fun(size_t threadid);
		   * \endcode
		   *
		   * This function runs an rmi barrier after termination
		   *
		   * @tparam the type of the member function.
		   * @param [in] member_fun the function to call.
		   */
		  template<typename MemberFunction>
		  void run_synchronous(MemberFunction member_fun) {
			shared_lvid_counter = 0;
			if (threads.size() <= 1) {
			  INCREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
			  ( (this)->*(member_fun))(0);
			  DECREMENT_EVENT(EVENT_ACTIVE_CPUS, 1);
			}
			else {
			  // launch the initialization threads
			  for(size_t i = 0; i < threads.size(); ++i) {
				boost::function<void(void)> invoke = boost::bind(member_fun, this, i);
				threads.launch(boost::bind(
					  &xadaptive_engine::thread_launch_wrapped_event_counter,
					  this,
					  invoke), i);
			  }
			}
			// Wait for all threads to finish
			threads.join();
			rmi.barrier();
		  } // end of run_synchronous
	  
		  // /**
		  //  * \brief Initialize all vertex programs by invoking
		  //  * \ref graphlab::ivertex_program::init on all vertices.
		  //  *
		  //  * @param thread_id the thread to run this as which determines
		  //  * which vertices to process.
		  //  */
		  // void initialize_vertex_programs(size_t thread_id);
	  
		  /**
		   * \brief Synchronize all message data.
		   *
		   * @param thread_id the thread to run this as which determines
		   * which vertices to process.
		   */
		  void exchange_messages(size_t thread_id);
	  
	  
		  /**
		   * \brief Invoke the \ref graphlab::ivertex_program::init function
		   * on all vertex programs that have inbound messages.
		   *
		   * @param thread_id the thread to run this as which determines
		   * which vertices to process.
		   */
		  void receive_messages(size_t thread_id);
	  
	  
		  /**
		   * \brief Execute the \ref graphlab::ivertex_program::gather function on all
		   * vertices that received messages for the edges specified by the
		   * \ref graphlab::ivertex_program::gather_edges.
		   *
		   * @param thread_id the thread to run this as which determines
		   * which vertices to process.
		   */
		  void execute_gathers(size_t thread_id);
	  
	  
	  
	  
		  /**
		   * \brief Execute the \ref graphlab::ivertex_program::apply function on all
		   * all vertices that received messages in this super-step (active).
		   *
		   * @param thread_id the thread to run this as which determines
		   * which vertices to process.
		   */
		  void execute_applys(size_t thread_id);
	  
		  /**
		   * \brief Execute the \ref graphlab::ivertex_program::scatter function on all
		   * vertices that received messages for the edges specified by the
		   * \ref graphlab::ivertex_program::scatter_edges.
		   *
		   * @param thread_id the thread to run this as which determines
		   * which vertices to process.
		   */
		  void execute_scatters(size_t thread_id);
	  
		  // Data Synchronization ===================================================
		  /**
		   * \brief Send the vertex program for the local vertex id to all
		   * of its mirrors.
		   *
		   * @param [in] lvid the vertex to sync.  This muster must be the
		   * master of that vertex.
		   */
		  void sync_vertex_program(lvid_type lvid, size_t thread_id);
	  
		  /**
		   * \brief Receive all incoming vertex programs and update the
		   * local mirrors.
		   *
		   * This function returns when there are no more incoming vertex
		   * programs and should be called after a flush of the vertex
		   * program exchange.
		   */
		  void recv_vertex_programs(const bool try_to_recv = false);
	  
		  /**
		   * \brief Send the vertex data for the local vertex id to all of
		   * its mirrors.
		   *
		   * @param [in] lvid the vertex to sync.  This machine must be the master
		   * of that vertex.
		   */
		  void sync_vertex_data(lvid_type lvid, size_t thread_id);
	  
		  /**
		   * \brief Receive all incoming vertex data and update the local
		   * mirrors.
		   *
		   * This function returns when there are no more incoming vertex
		   * data and should be called after a flush of the vertex data
		   * exchange.
		   */
		  void recv_vertex_data(const bool try_to_recv = false);
	  
		  /**
		   * \brief Send the gather value for the vertex id to its master.
		   *
		   * @param [in] lvid the vertex to send the gather value to
		   * @param [in] accum the locally computed gather value.
		   */
		  void sync_gather(lvid_type lvid, const gather_type& accum,
						   size_t thread_id);
	  
	  
		  /**
		   * \brief Receive the gather values from the buffered exchange.
		   *
		   * This function returns when there is nothing left in the
		   * buffered exchange and should be called after the buffered
		   * exchange has been flushed
		   */
		  void recv_gathers(const bool try_to_recv = false);
	  
		  /**
		   * \brief Send the accumulated message for the local vertex to its
		   * master.
		   *
		   * @param [in] lvid the vertex to send
		   */
		  void sync_message(lvid_type lvid, const size_t thread_id);
	  
		  /**
		   * \brief Receive the messages from the buffered exchange.
		   *
		   * This function returns when there is nothing left in the
		   * buffered exchange and should be called after the buffered
		   * exchange has been flushed
		   */
		  void recv_messages(const bool try_to_recv = false);
	  
	  
		}; // end of class


  	/*
  	*	==================== template  methods ===================================
  	*/
  	
  /**
   * Constructs an synchronous distributed engine.
   * The number of threads to create are read from
   * opts::get_ncpus().
   *
   * Valid engine options (graphlab_options::get_engine_args()):
   * \arg \c max_iterations Sets the maximum number of iterations the
   * engine will run for.
   * \arg \c use_cache If set to true, partial gathers are cached.
   * See \ref gather_caching to understand the behavior of the
   * gather caching model and how it may be used to accelerate program
   * performance.
   *
   * \param dc Distributed controller to associate with
   * \param graph The graph to schedule over. The graph must be fully
   *              constructed and finalized.
   * \param opts A graphlab_options object containing options and parameters
   *             for the engine.
   */
  template<typename VertexProgram>
  xadaptive_engine<VertexProgram>::
  xadaptive_engine(distributed_control &dc,
                     graph_type& graph,
                     const graphlab_options& opts) :
    rmi(dc, this), graph(graph),
    threads(opts.get_ncpus()),
    thread_barrier(opts.get_ncpus()),
    max_iterations(-1), snapshot_interval(-1), iteration_counter(0),
    timeout(0), sched_allv(false),
    vprog_exchange(dc, opts.get_ncpus(), 64 * 1024),
    vdata_exchange(dc, opts.get_ncpus(), 64 * 1024),
    gather_exchange(dc, opts.get_ncpus(), 64 * 1024),
    message_exchange(dc, opts.get_ncpus(), 64 * 1024),
    aggregator(dc, graph, new context_type(*this, graph)),
	//xie insert: async engine init
	scheduler_ptr(NULL), started(false), engine_start_time(timer::approx_time_seconds()), force_stop(false) {

	// xie insert: should set start mode, it decides how to receive the start signal
	current_engine = X_SYNC;
	running_mode= X_ADAPTIVE;
	has_max_iterations = false;	
	 
    // Process any additional options
    std::vector<std::string> keys = opts.get_engine_args().get_option_keys();
    per_thread_compute_time.resize(opts.get_ncpus());
    use_cache = false;

	//xie insert set async engine
	nfibers = 3000;
    stacksize = 16384;
    factorized_consistency = true;
    timed_termination = (size_t)(-1);
    termination_reason = execution_status::UNSET;
	X_A_Threshold_low = 500000000;//0.01;
	X_A_Threshold_hig = 500000000;//0.015;
	X_S_Min_Iters = 5;
	X_S_Increase_Rate = -(0.00001);
	X_S_Sampled_Iters = 10;
	X_A_Join_Rate = 0.4;
	X_A_Delay_Counter = 20;
	X_A_Delay = 200;
	A_Sampled_Iters = 5;
	T_SAMPLE = 1000;

	//xie insert : set sync & async opts
	xset_options(opts);
	
    if (snapshot_interval >= 0 && snapshot_path.length() == 0) {
      logstream(LOG_FATAL)
        << "Snapshot interval specified, but no snapshot path" << std::endl;
    }
    INITIALIZE_EVENT_LOG(dc);
    ADD_CUMULATIVE_EVENT(EVENT_APPLIES, "Applies", "Calls");
    ADD_CUMULATIVE_EVENT(EVENT_GATHERS , "Gathers", "Calls");
    ADD_CUMULATIVE_EVENT(EVENT_SCATTERS , "Scatters", "Calls");
    ADD_INSTANTANEOUS_EVENT(EVENT_ACTIVE_CPUS, "Active Threads", "Threads");
    graph.finalize();
    init();

	//xie insert: async init
	scheduler_ptr->set_num_vertices(graph.num_local_vertices());
    xmessages.resize(graph.num_local_vertices());
    vertexlocks.resize(graph.num_local_vertices());
    program_running.resize(graph.num_local_vertices());
    hasnext.resize(graph.num_local_vertices());
    if (use_cache) {
        gather_cache.resize(graph.num_local_vertices(), gather_type());
        has_cache.resize(graph.num_local_vertices());
        has_cache.clear();
    }
    if (!factorized_consistency) {
        cm_handles.resize(graph.num_local_vertices());
    }
    rmi.barrier();
  } // end of synchronous engine


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>:: init() {
    resize();
    // Clear up
    force_abort = false;
    iteration_counter = 0;
    completed_applys = 0;
    has_message.clear();
    has_gather_accum.clear();
    has_cache.clear();
    active_superstep.clear();
    active_minorstep.clear();
	//xie insert clear();
	asy_start_active_v.clear();
	first_time_start = true;
	each_iteration_signal = 0;
	each_iteration_nmsg = 0;
  }


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>:: resize() {
    memory_info::log_usage("Before Engine Initialization");
    // Allocate vertex locks and vertex programs
    vlocks.resize(graph.num_local_vertices());
    vertex_programs.resize(graph.num_local_vertices());
    // allocate the edge locks
    //elocks.resize(graph.num_local_edges());
    // Allocate messages and message bitset
    messages.resize(graph.num_local_vertices(), message_type());
    has_message.resize(graph.num_local_vertices());
	//xie insert resize();
	//asy_now_active_v.resize(graph.num_local_vertices());
	asy_start_active_v.resize(graph.num_local_vertices());
	
    // Allocate gather accumulators and accumulator bitset
    gather_accum.resize(graph.num_local_vertices(), gather_type());
    has_gather_accum.resize(graph.num_local_vertices());

    // If caching is used then allocate cache data-structures
    if (use_cache) {
      gather_cache.resize(graph.num_local_vertices(), gather_type());
      has_cache.resize(graph.num_local_vertices());
    }
    // Allocate bitset to track active vertices on each bitset.
    active_superstep.resize(graph.num_local_vertices());
    active_minorstep.resize(graph.num_local_vertices());

    // Print memory usage after initialization
    memory_info::log_usage("After Engine Initialization");
  }


  template<typename VertexProgram>
  typename xadaptive_engine<VertexProgram>::aggregator_type*
  xadaptive_engine<VertexProgram>::get_aggregator() {
    return &aggregator;
  } // end of get_aggregator



  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::internal_stop() {
  	// xie insert: SYNC
    if(current_engine==X_SYNC){
	    for (size_t i = 0; i < rmi.numprocs(); ++i)
	      rmi.remote_call(i, &xadaptive_engine<VertexProgram>::rpc_stop);
    }
	// xie insert ASYNC
	else if(current_engine==X_ASYNC){
		xinternal_stop();
	}
  } // end of internal_stop

  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::rpc_stop() {
    force_abort = true;
  } // end of rpc_stop


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  signal(vertex_id_type gvid, const message_type& message) {
  	// xie insert SYNC
  	if(current_engine==X_SYNC){
	    if (vlocks.size() != graph.num_local_vertices())
	      resize();
	    rmi.barrier();
	    internal_signal_rpc(gvid, message);
	    rmi.barrier();
  	}
	// xie insert ASYNC
	else {
	  rmi.barrier();
      xinternal_signal_gvid(gvid, message);
      rmi.barrier();
	}
  } // end of signal

  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  signal_all(const message_type& message, const std::string& order) {
	// xie insert: SYNC
	if(current_engine==X_SYNC){
	    if (vlocks.size() != graph.num_local_vertices())
	      resize();
	    for(lvid_type lvid = 0; lvid < graph.num_local_vertices(); ++lvid) {
	      if(graph.l_is_master(lvid)) {
	        internal_signal(vertex_type(graph.l_vertex(lvid)), message);
	      }
	    }
	}
	// xie insert: ASYNC
	else{
		xsignal_all(message, order);
	}
  } // end of signal all


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  signal_vset(const vertex_set& vset,
             const message_type& message, const std::string& order) {
    // xie insert: SYNC
    if(current_engine==X_SYNC){
	    if (vlocks.size() != graph.num_local_vertices())
	      resize();
	    for(lvid_type lvid = 0; lvid < graph.num_local_vertices(); ++lvid) {
	      if(graph.l_is_master(lvid) && vset.l_contains(lvid)) {
	        internal_signal(vertex_type(graph.l_vertex(lvid)), message);
	      }
	    }
    }
	// xie insert: ASYNC
	else {
		xsignal_vset(vset, message, order);
	}
  } // end of signal all

  //xie insert
  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  s_inner_signal_vset(const std::string& order) {
    // xie insert: SYNC mode insert intermediate active nodes
	if (vlocks.size() != graph.num_local_vertices())
	   resize();
	for(lvid_type lvid = 0; lvid < graph.num_local_vertices(); ++lvid) {
	   if(graph.l_is_master(lvid) && next_mode_active_vertex.get(lvid)) {
	   	  message_type msg;
		  xmessages.get(lvid,msg);
	      internal_signal(vertex_type(graph.l_vertex(lvid)), msg);
	   }
	}
    
  } // end of signal all

  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  internal_signal(const vertex_type& vertex,
                  const message_type& message) {
    // xie insert: SYNC
    if(current_engine==X_SYNC){
		const lvid_type lvid = vertex.local_id();
		
		vlocks[lvid].lock();
		
		if( has_message.get(lvid) ) {
			messages[lvid] += message;
		} else {
		    messages[lvid] = message;
		    has_message.set_bit(lvid);

			//xie insert;
			asy_start_active_v.set_bit(lvid);
		}
		vlocks[lvid].unlock();
    	}
	// xie insert: ASYNC
	else{
		xinternal_signal(vertex,message);
	}
  } // end of internal_signal


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  internal_signal_broadcast(vertex_id_type gvid, const message_type& message) {
  	// xie insert: SYNC
    if(current_engine==X_SYNC){
	    for (size_t i = 0; i < rmi.numprocs(); ++i) {
	      if(i == rmi.procid()) internal_signal_rpc(gvid, message);
	      else rmi.remote_call(i, &xadaptive_engine<VertexProgram>::internal_signal_rpc,
	                          gvid, message);
	    }
	}
	// xie insert: ASYNC
	else {
		xinternal_signal_broadcast(gvid,message);
	}
  } // end of internal_signal_broadcast


  //xie insert: ASYNC engine does not support this.
  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  internal_signal_rpc(vertex_id_type gvid,
                      const message_type& message) {
    if (graph.is_master(gvid)) {
      internal_signal(graph.vertex(gvid), message);
    }
	
  } // end of internal_signal_rpc





  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  internal_post_delta(const vertex_type& vertex, const gather_type& delta) {
    const bool caching_enabled = !gather_cache.empty();
    if(caching_enabled) {
      //xie insert: confirm the implementations in cache are the same in both SYNC and ASYNC 
	  //ASSERT_TRUE(use_cache);
      const lvid_type lvid = vertex.local_id();
      vlocks[lvid].lock();
      if( has_cache.get(lvid) ) {
        gather_cache[lvid] += delta;
      } else {
        // You cannot add a delta to an empty cache.  A complete
        // gather must have been run.
        // gather_cache[lvid] = delta;
        // has_cache.set_bit(lvid);
      }
      vlocks[lvid].unlock();
    }
	else{
		//xie insert: confirm the implementations in cache are the same in both SYNC and ASYNC 
		//ASSERT_FALSE(use_cache);
	}
  } // end of post_delta


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  internal_clear_gather_cache(const vertex_type& vertex) {
    const bool caching_enabled = !gather_cache.empty();
    const lvid_type lvid = vertex.local_id();
    if(caching_enabled && has_cache.get(lvid)) {
      vlocks[lvid].lock();
      gather_cache[lvid] = gather_type();
      has_cache.clear_bit(lvid);
      vlocks[lvid].unlock();
    }
  } // end of clear_gather_cache



  template<typename VertexProgram>
  size_t xadaptive_engine<VertexProgram>::
  num_updates() const { return completed_applys.value+programs_executed.value; }

  template<typename VertexProgram>
  float xadaptive_engine<VertexProgram>::
  elapsed_seconds() const { return timer::approx_time_seconds() - start_time; }

  template<typename VertexProgram>
  int xadaptive_engine<VertexProgram>::
  iteration() const { return iteration_counter; }



  template<typename VertexProgram>
  size_t xadaptive_engine<VertexProgram>::total_memory_usage() const {
    size_t allocated_memory = memory_info::allocated_bytes();
    rmi.all_reduce(allocated_memory);
    return allocated_memory;
  } // compute the total memory usage of the GraphLab system

  template<typename VertexProgram> execution_status::status_enum
	xadaptive_engine<VertexProgram>::sstart() {
	  //xie insert
	  current_engine = X_SYNC;
	  if(first_time_start)
	  	first_time_start = false;
	  else{
	  	//signal 
	  	s_inner_signal_vset();
		xmessages.clear();
	  	}
	  xmessages.clear();

	  float start_this_turn;		// used for set least execution time
	  size_t total_act = 0;
	  
	  // xie insert: SYNC engine compution start
	  aggregator.start();
	  rmi.barrier();
  
	  /* xie delete: temporary not support it 
	  if (snapshot_interval == 0) {
		graph.save_binary(snapshot_path);
	  }*/
  
	  float last_print = -5;
	  if (rmi.procid() == 0) {
		logstream(LOG_EMPH) << rmi.procid()<<": Iteration counter will only output every 5 seconds. Switch overhead "
						  << (globaltimer.current_time_millis() -countoverhead) << std::endl;
	  }


	  double timelast = globaltimer.current_time_millis();
	  size_t lastactive = 0;
	  double tmpconst = -1;
	  double last_thro = 0;
	  double lasttime=timelast;
	  
	  // Program Main loop ====================================================
	  while(iteration_counter <= max_iterations && !force_abort ) {
		//double time_estart = globaltimer.current_time_millis();
		
		// Check first to see if we are out of time
		if(timeout != 0 && timeout < elapsed_seconds()) {
		  termination_reason = execution_status::TIMEOUT;
		  break;
		}
  		
		bool print_this_round = (elapsed_seconds() - last_print) >= 5;
  
		if(rmi.procid() == 0 && print_this_round) {
		  logstream(LOG_EMPH)
			<< rmi.procid() << ": Starting iteration: " << iteration_counter
			<< " lastact "<<lastactive
			<< std::endl;
		  last_print = elapsed_seconds();
		}
		// Reset Active vertices ----------------------------------------------
		// Clear the active super-step and minor-step bits which will
		// be set upon receiving messages
		active_superstep.clear(); active_minorstep.clear();
		has_gather_accum.clear();
		rmi.barrier();
  
		// Exchange Messages --------------------------------------------------
		// Exchange any messages in the local message vectors
		// if (rmi.procid() == 0) std::cout << "Exchange messages..." << std::endl;
		run_synchronous( &xadaptive_engine::exchange_messages );
		/**
		 * Post conditions:
		 *	 1) only master vertices have messages
		 */
  
		// Receive Messages ---------------------------------------------------
		// Receive messages to master vertices and then synchronize
		// vertex programs with mirrors if gather is required
		//
  
		// if (rmi.procid() == 0) std::cout << "Receive messages..." << std::endl;
		num_active_vertices = 0;
		//xie insert
		num_active_mirrors = 0;
		//double time_rstart = globaltimer.current_time_millis();
		
		run_synchronous( &xadaptive_engine::receive_messages );
		if (sched_allv) {
		  active_minorstep.fill();
		}
		has_message.clear();
		/**
		 * Post conditions:
		 *	 1) there are no messages remaining
		 *	 2) All masters that received messages have their
		 *		active_superstep bit set
		 *	 3) All masters and mirrors that are to participate in the
		 *		next gather phases have their active_minorstep bit
		 *		set.
		 *	 4) num_active_vertices is the number of vertices that
		 *		received messages.
		 */
  
		// Check termination condition	---------------------------------------
		size_t total_active_vertices = num_active_vertices;
		
		rmi.all_reduce(total_active_vertices);

		if(total_active_vertices == 0 ) {
		  termination_reason = execution_status::TASK_DEPLETION;
		  break;
		}

		double time_rend = globaltimer.current_time_millis();
  		//xie insert
  		//================================================================
		//xie insert: when iteration_counter==max_iterations, get the active vertex set and break;
		//size_t total_signal_number = each_iteration_signal;
		//size_t total_msg_number = each_iteration_nmsg;
		//size_t total_active_mirrors = num_active_mirrors;
		//rmi.all_reduce(total_signal_number);
		//rmi.all_reduce(total_msg_number);
		//rmi.all_reduce(total_active_mirrors);
		//each_iteration_signal = 0;
		//each_iteration_nmsg = 0;
		
	
		/*if (rmi.procid() == 0 )
		{
			double this_iter_time = globaltimer.current_time_millis()-timelast;
			logstream(LOG_EMPH)<< rmi.procid() << ":iter " << iteration_counter
			//<<" , max_iterations "<<max_iterations
			<<" ,total_iter_msg "<<total_msg_number
			//<<" , each_iter_sig "<<each_iteration_signal
			//<<" , total_sig_num "<<total_signal_number
			//<<" , local_active_vertex "<<local_active_vertices
			<<" ,total_act_ver "<<total_active_vertices
			<<" ,total_act_mir "<<total_active_mirrors
			<<" ,thro_last "<<lastactive/3.0/this_iter_time
			<<" ,At "<<this_iter_time //<<" ,e: "<<time_rstart-time_estart<<" ,r: "<<time_rend-time_estart
			<<std::endl;
			lastactive = total_active_vertices;
			total_act+=total_active_vertices;
		}*/
	    //timelast = globaltimer.current_time_millis();

		
		// clear counters
		//each_iteration_signal = 0;
		//each_iteration_nmsg = 0;

		float fac = 1;
		float avg_inc_rate = 1;


		double thro_now=999999;
		if(iteration_counter==0){
			for(int i=0; i<11;i++){
		  		avg_line[i] = total_active_vertices;
				active[i] = total_active_vertices;
		  	}
			lastactive = total_active_vertices;
		}
		else //if(!has_max_iterations)
		{
			//fac = ((float)total_active_vertices)/graph.num_vertices();
			size_t now = iteration_counter%11; 
			active[now] = total_active_vertices;//fac;
			avg_line[now] = avg_line[(iteration_counter+10)%11]+(active[now]-active[(iteration_counter-X_S_Sampled_Iters+11)%11])/X_S_Sampled_Iters;
			avg_inc_rate = avg_line[now]-avg_line[(iteration_counter+10)%11];
				
			double this_iter_time = globaltimer.current_time_millis()-timelast;
			double thro = lastactive/this_iter_time/rmi.numprocs();
			if(tmpconst<0)
				tmpconst = thro/lastactive*3;
			else tmpconst = (tmpconst+ thro/lastactive)/2;
			last_thro = thro;
			total_act+=total_active_vertices;
			thro_now = tmpconst*total_active_vertices*rate_AvsS;
			if (rmi.procid() == 0 )
				logstream(LOG_EMPH)<< rmi.procid() << ":iter "<< iteration_counter
				<<" , thro "<<thro
				<<" , act "<<lastactive
				<<" , time "<<this_iter_time
				<<" , t_now "<<thro_now
				<<" , const "<<tmpconst
				<<std::endl;
			lastactive = total_active_vertices;
			timelast = globaltimer.current_time_millis();

			if(running_mode==X_SAMPLE)
			{
				double now = globaltimer.current_time_millis();
				if(now-lasttime>2000){
					double thros = (total_act-total_active_vertices)/(now-lasttime)/rmi.numprocs();
					if (rmi.procid() == 0 )
						logstream(LOG_EMPH)<< rmi.procid() << ": thro "<< thros
							<<" ,time "<<(now-lasttime)
							<<" ,nowtime "<<globaltimer.current_time_millis()
							<<std::endl;
					total_act=total_active_vertices;
					lasttime = globaltimer.current_time_millis();
				}
			}
			//else 
			if(running_mode==X_MANUAL){
				if(iteration_counter>=switch_iter){
					if (rmi.procid() == 0)
						logstream(LOG_EMPH)<< rmi.procid() << ":iter "<< iteration_counter//<<" ,fac "<<fac
							<<" ,tol_active "<<total_active_vertices//<<" X_S_Increase_Rate "<<X_S_Increase_Rate
							<<" ,avg_inc "<<avg_inc_rate
							<<rmi.numprocs()<<std::endl;

					countoverhead = globaltimer.current_time_millis();
					
					// if iteration_counter==0, next_mode_active_vertex has been set to initial signal set.
					if(iteration_counter==0)
						next_mode_active_vertex = asy_start_active_v;
					else
						next_mode_active_vertex = active_superstep;
					
					termination_reason = execution_status::MODE_SWITCH;
					break;
				}
			}\
			else if((avg_inc_rate<0)&&(thro_now<=thro_A)){
					if (rmi.procid() == 0 )
						logstream(LOG_EMPH)<< rmi.procid() << ":iter "<< iteration_counter
						<<" ,nor_s_thro "<<thro_now
						<<" ,thro_A "<<thro_A
						<<" ,act "<<total_active_vertices	
						<<" ,avg_inc "<<avg_inc_rate
						<<std::endl;					
					countoverhead = globaltimer.current_time_millis();

					if(iteration_counter==0)
						next_mode_active_vertex = asy_start_active_v;
					else
						next_mode_active_vertex = active_superstep;
					
					termination_reason = execution_status::MODE_SWITCH;
					break;
				
			}
		
		}
		

		
		
  		//================================================================
  		//xie insert end
  		
		
		// Execute gather operations-------------------------------------------
		// Execute the gather operation for all vertices that are active
		// in this minor-step (active-minorstep bit set).
		// if (rmi.procid() == 0) std::cout << "Gathering..." << std::endl;
		run_synchronous( &xadaptive_engine::execute_gathers );
		// Clear the minor step bit since only super-step vertices
		// (only master vertices are required to participate in the
		// apply step)
		active_minorstep.clear(); // rmi.barrier();
		/**
		 * Post conditions:
		 *	 1) gather_accum for all master vertices contains the
		 *		result of all the gathers (even if they are drawn from
		 *		cache)
		 *	 2) No minor-step bits are set
		 */
  
		// Execute Apply Operations -------------------------------------------
		// Run the apply function on all active vertices
		// if (rmi.procid() == 0) std::cout << "Applying..." << std::endl;
	  run_synchronous( &xadaptive_engine::execute_applys );
		/**
		 * Post conditions:
		 *	 1) any changes to the vertex data have been synchronized
		 *		with all mirrors.
		 *	 2) all gather accumulators have been cleared
		 *	 3) If a vertex program is participating in the scatter
		 *		phase its minor-step bit has been set to active (both
		 *		masters and mirrors) and the vertex program has been
		 *		synchronized with the mirrors.
		 */
  
  
	  // Execute Scatter Operations -----------------------------------------
	  // Execute each of the scatters on all minor-step active vertices.
	  run_synchronous( &xadaptive_engine::execute_scatters );
	  /**
		 * Post conditions:
		 *	 1) NONE
		 */
	  	if(rmi.procid() == 0 && print_this_round)
		  logstream(LOG_EMPH) << "\t Running Aggregators" << std::endl;
		// probe the aggregator
		aggregator.tick_synchronous();
  
		++iteration_counter;
  
		if (snapshot_interval > 0 && iteration_counter % snapshot_interval == 0) {
		  graph.save_binary(snapshot_path);
		}
	  }
  
	  rmi.full_barrier();
	  // Stop the aggregator
	  aggregator.stop();

	  if (rmi.procid() == 0) {
      logstream(LOG_EMPH) << iteration_counter
                        << " iterations completed. tol act "<<total_act<< std::endl;
	  	}
                        
	  // return the final reason for termination
	  return termination_reason;
	} // end of start
	

  template<typename VertexProgram> execution_status::status_enum
  xadaptive_engine<VertexProgram>::start() {
    if (vlocks.size() != graph.num_local_vertices())
      resize();
    completed_applys = 0;
	rmi.barrier();

    // Initialization code ==================================================
    // Reset event log counters?
    // Start the timer
    start_time = timer::approx_time_seconds();
    iteration_counter = 0;
    force_abort = false;
    execution_status::status_enum termination_reason =
      execution_status::UNSET;

	globaltimer.start();
	x_start_time_m = globaltimer.current_time_millis();	// start time
	if(current_engine == X_SYNC)
		//begin with SYNC engine
    	termination_reason = sstart();
	else 
		termination_reason = xstart();


	while(termination_reason==execution_status::MODE_SWITCH){
		//try to switch engines
		if(current_engine == X_SYNC){
			//switch to ASYNC
			scheduler_ptr->set_num_vertices(graph.num_local_vertices());
		    vertexlocks.clear();
		    program_running.clear();
		    hasnext.clear();
		    if (use_cache)
		        has_cache.clear();
		    if (!factorized_consistency) 
		        cm_handles.clear();
	
			if (rmi.procid() == 0)
				logstream(LOG_EMPH) << "Switch to ASYNC. At "<<(globaltimer.current_time_millis()-x_start_time_m)<<std::endl;
			termination_reason = xstart();
		}
		else{
			//switch to SYNC
			if (rmi.procid() == 0)
				logstream(LOG_EMPH) << "Switch to SYNC. At "<<(globaltimer.current_time_millis()-x_start_time_m)<<std::endl;
			force_abort = false;
		    iteration_counter = 0;
		    completed_applys = 0;
		    has_message.clear();
		    has_gather_accum.clear();
		    has_cache.clear();			//xie insert: haven't implement cache support
		    active_superstep.clear();
		    active_minorstep.clear();
			each_iteration_signal = 0;
			each_iteration_nmsg = 0;
			termination_reason = sstart();
	
			}
		}

	//handle termination

	/*if (rmi.procid() == 0) {
      logstream(LOG_EMPH) << iteration_counter
                        << " iterations completed." << std::endl;
    	}*/
    	
    // Final barrier to ensure that all engines terminate at the same time
   /* double total_compute_time = 0;
    for (size_t i = 0;i < per_thread_compute_time.size(); ++i) {
      total_compute_time += per_thread_compute_time[i];
    }
    std::vector<double> all_compute_time_vec(rmi.numprocs());
    all_compute_time_vec[rmi.procid()] = total_compute_time;
    rmi.all_gather(all_compute_time_vec);

    size_t global_completed = completed_applys;
    rmi.all_reduce(global_completed);
    completed_applys = global_completed;
    rmi.cout() << "Updates: " << completed_applys.value << "\n";
    if (rmi.procid() == 0) {
      logstream(LOG_INFO) << "Compute Balance: ";
      for (size_t i = 0;i < all_compute_time_vec.size(); ++i) {
        logstream(LOG_INFO) << all_compute_time_vec[i] << " ";
      }
      logstream(LOG_INFO) << std::endl;
    }
    rmi.full_barrier();
    // Stop the aggregator
    aggregator.stop();*/
    // return the final reason for termination
    return termination_reason;
    
  } // end of start



  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  exchange_messages(const size_t thread_id) {
    context_type context(*this, graph);
    const bool TRY_TO_RECV = true;
    const size_t TRY_RECV_MOD = 100;
    size_t vcount = 0;
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = has_message.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;
        // if the vertex is not local and has a message send the
        // message and clear the bit
        if(!graph.l_is_master(lvid)) {
          sync_message(lvid, thread_id);
          has_message.clear_bit(lvid);
          // clear the message to save memory
          messages[lvid] = message_type();
        }
        if(++vcount % TRY_RECV_MOD == 0) recv_messages(TRY_TO_RECV);
      }
    } // end of loop over vertices to send messages
    message_exchange.partial_flush(thread_id);
    // Finish sending and receiving all messages
    thread_barrier.wait();
    if(thread_id == 0) message_exchange.flush();
    thread_barrier.wait();
    recv_messages();
  } // end of exchange_messages



  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  receive_messages(const size_t thread_id) {
    context_type context(*this, graph);
    const bool TRY_TO_RECV = true;
    const size_t TRY_RECV_MOD = 100;
    size_t vcount = 0;
    size_t nactive_inc = 0;
	//xie insert
	size_t nactive_mirrors = 0;
	
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit

    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = has_message.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));

      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // if this is the master of lvid and we have a message
        if(graph.l_is_master(lvid)) {
          // The vertex becomes active for this superstep
          active_superstep.set_bit(lvid); 
          ++nactive_inc;
          // Pass the message to the vertex program
          vertex_type vertex = vertex_type(graph.l_vertex(lvid));
          vertex_programs[lvid].init(context, vertex, messages[lvid]);
          // clear the message to save memory
          //messages[lvid] = message_type();	xie modify
		  //xie insert
		  nactive_mirrors += graph.l_vertex(lvid).num_mirrors();
	
		  if (sched_allv) continue;
          // Determine if the gather should be run
          const vertex_program_type& const_vprog = vertex_programs[lvid];
          const vertex_type const_vertex = vertex;
          if(const_vprog.gather_edges(context, const_vertex) !=
              graphlab::NO_EDGES) {
            active_minorstep.set_bit(lvid);
            sync_vertex_program(lvid, thread_id);
          }
        }
        if(++vcount % TRY_RECV_MOD == 0) recv_vertex_programs(TRY_TO_RECV);
      }
    }

    num_active_vertices += nactive_inc;
	//xie insert
	num_active_mirrors += nactive_mirrors;
	
    vprog_exchange.partial_flush(thread_id);
    // Flush the buffer and finish receiving any remaining vertex
    // programs.
    thread_barrier.wait();
    if(thread_id == 0) {
      vprog_exchange.flush();
    }
    thread_barrier.wait();

    recv_vertex_programs();

  } // end of receive messages


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  execute_gathers(const size_t thread_id) {
    context_type context(*this, graph);
    const bool TRY_TO_RECV = true;
    const size_t TRY_RECV_MOD = 1000;
    size_t vcount = 0;
    const bool caching_enabled = !gather_cache.empty();
    timer ti;

    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // a word-size = 64 bit

    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_minorstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));

      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        bool accum_is_set = false;
        gather_type accum = gather_type();
        // if caching is enabled and we have a cache entry then use
        // that as the accum
        if( caching_enabled && has_cache.get(lvid) ) {
          accum = gather_cache[lvid];
          accum_is_set = true;
        } else {
          // recompute the local contribution to the gather
          const vertex_program_type& vprog = vertex_programs[lvid];
          local_vertex_type local_vertex = graph.l_vertex(lvid);
          const vertex_type vertex(local_vertex);
          const edge_dir_type gather_dir = vprog.gather_edges(context, vertex);
          // Loop over in edges
          size_t edges_touched = 0;
          vprog.pre_local_gather(accum);
          if(gather_dir == IN_EDGES || gather_dir == ALL_EDGES) {
            foreach(local_edge_type local_edge, local_vertex.in_edges()) {
              edge_type edge(local_edge);
              // elocks[local_edge.id()].lock();
              if(accum_is_set) { // \todo hint likely
                accum += vprog.gather(context, vertex, edge);
              } else {
                accum = vprog.gather(context, vertex, edge);
                accum_is_set = true;
              }
              ++edges_touched;
              // elocks[local_edge.id()].unlock();
            }
          } // end of if in_edges/all_edges
            // Loop over out edges
          if(gather_dir == OUT_EDGES || gather_dir == ALL_EDGES) {
            foreach(local_edge_type local_edge, local_vertex.out_edges()) {
              edge_type edge(local_edge);
              // elocks[local_edge.id()].lock();
              if(accum_is_set) { // \todo hint likely
                accum += vprog.gather(context, vertex, edge);
              } else {
                accum = vprog.gather(context, vertex, edge);
                accum_is_set = true;
              }
              // elocks[local_edge.id()].unlock();
              ++edges_touched;
            }
            INCREMENT_EVENT(EVENT_GATHERS, edges_touched);
          } // end of if out_edges/all_edges
          vprog.post_local_gather(accum);
          // If caching is enabled then save the accumulator to the
          // cache for future iterations.  Note that it is possible
          // that the accumulator was never set in which case we are
          // effectively "zeroing out" the cache.
          if(caching_enabled && accum_is_set) {
            gather_cache[lvid] = accum; has_cache.set_bit(lvid);
          } // end of if caching enabled
        }
        // If the accum contains a value for the local gather we put
        // that estimate in the gather exchange.
        if(accum_is_set) sync_gather(lvid, accum, thread_id);
        if(!graph.l_is_master(lvid)) {
          // if this is not the master clear the vertex program
          vertex_programs[lvid] = vertex_program_type();
        }

        // try to recv gathers if there are any in the buffer
        if(++vcount % TRY_RECV_MOD == 0) recv_gathers(TRY_TO_RECV);
      }
    } // end of loop over vertices to compute gather accumulators
    per_thread_compute_time[thread_id] += ti.current_time();
    gather_exchange.partial_flush(thread_id);
      // Finish sending and receiving all gather operations
    thread_barrier.wait();
    if(thread_id == 0) gather_exchange.flush();
    thread_barrier.wait();
    recv_gathers();
  } // end of execute_gathers


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  execute_applys(const size_t thread_id) {
    context_type context(*this, graph);
    const bool TRY_TO_RECV = true;
    const size_t TRY_RECV_MOD = 1000;
    size_t vcount = 0;
    timer ti;

    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset;  // allocate a word size = 64bits
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_superstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        // Only master vertices can be active in a super-step
        ASSERT_TRUE(graph.l_is_master(lvid));
		//xie insert
		messages[lvid] = message_type();
		//xie insert end
        vertex_type vertex(graph.l_vertex(lvid));
        // Get the local accumulator.  Note that it is possible that
        // the gather_accum was not set during the gather.
        const gather_type& accum = gather_accum[lvid];
        INCREMENT_EVENT(EVENT_APPLIES, 1);
        vertex_programs[lvid].apply(context, vertex, accum);
        // record an apply as a completed task
        ++completed_applys;
        // Clear the accumulator to save some memory
        gather_accum[lvid] = gather_type();
        // synchronize the changed vertex data with all mirrors
        sync_vertex_data(lvid, thread_id);
        // determine if a scatter operation is needed
        const vertex_program_type& const_vprog = vertex_programs[lvid];
        const vertex_type const_vertex = vertex;
		
        if(const_vprog.scatter_edges(context, const_vertex) !=
           graphlab::NO_EDGES) {
          active_minorstep.set_bit(lvid);
          sync_vertex_program(lvid, thread_id);
        } else { // we are done so clear the vertex program
          vertex_programs[lvid] = vertex_program_type();
        }
      // try to receive vertex data
        if(++vcount % TRY_RECV_MOD == 0) {
          recv_vertex_programs(TRY_TO_RECV);
          recv_vertex_data(TRY_TO_RECV);
        }
      }
    } // end of loop over vertices to run apply

    per_thread_compute_time[thread_id] += ti.current_time();
    vprog_exchange.partial_flush(thread_id);
    vdata_exchange.partial_flush(thread_id);
      // Finish sending and receiving all changes due to apply operations
    thread_barrier.wait();
    if(thread_id == 0) { vprog_exchange.flush(); vdata_exchange.flush(); }
    thread_barrier.wait();
    recv_vertex_programs();
    recv_vertex_data();

  } // end of execute_applys




  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  execute_scatters(const size_t thread_id) {
    context_type context(*this, graph);
    timer ti;
    fixed_dense_bitset<8 * sizeof(size_t)> local_bitset; // allocate a word size = 64 bits

	//xie insert
	size_t local_signal = 0;
	
    while (1) {
      // increment by a word at a time
      lvid_type lvid_block_start =
                  shared_lvid_counter.inc_ret_last(8 * sizeof(size_t));
      if (lvid_block_start >= graph.num_local_vertices()) break;
      // get the bit field from has_message
      size_t lvid_bit_block = active_minorstep.containing_word(lvid_block_start);
      if (lvid_bit_block == 0) continue;
      // initialize a word sized bitfield
      local_bitset.clear();
      local_bitset.initialize_from_mem(&lvid_bit_block, sizeof(size_t));
      foreach(size_t lvid_block_offset, local_bitset) {
        lvid_type lvid = lvid_block_start + lvid_block_offset;
        if (lvid >= graph.num_local_vertices()) break;

        const vertex_program_type& vprog = vertex_programs[lvid];
        local_vertex_type local_vertex = graph.l_vertex(lvid);
        const vertex_type vertex(local_vertex);
        const edge_dir_type scatter_dir = vprog.scatter_edges(context, vertex);
				size_t edges_touched = 0;
        // Loop over in edges
        if(scatter_dir == IN_EDGES || scatter_dir == ALL_EDGES) {
		  //xie insert
		  local_signal += local_vertex.num_out_edges();
			
          foreach(local_edge_type local_edge, local_vertex.in_edges()) {
            edge_type edge(local_edge);
            // elocks[local_edge.id()].lock();
            vprog.scatter(context, vertex, edge);
            // elocks[local_edge.id()].unlock();
          }
					++edges_touched;
        } // end of if in_edges/all_edges
        // Loop over out edges
        if(scatter_dir == OUT_EDGES || scatter_dir == ALL_EDGES) {
		  //xie insert
		  local_signal += local_vertex.num_out_edges();
		  
          foreach(local_edge_type local_edge, local_vertex.out_edges()) {
            edge_type edge(local_edge);
            // elocks[local_edge.id()].lock();
            vprog.scatter(context, vertex, edge);
            // elocks[local_edge.id()].unlock();
          }
					++edges_touched;
        } // end of if out_edges/all_edges
				INCREMENT_EVENT(EVENT_SCATTERS, edges_touched);
        // Clear the vertex program
        vertex_programs[lvid] = vertex_program_type();
      } // end of if active on this minor step
    } // end of loop over vertices to complete scatter operation

	//xie insert
	each_iteration_signal += local_signal;
	
    per_thread_compute_time[thread_id] += ti.current_time();
  } // end of execute_scatters



  // Data Synchronization ===================================================
  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  sync_vertex_program(lvid_type lvid, const size_t thread_id) {
    ASSERT_TRUE(graph.l_is_master(lvid));
    const vertex_id_type vid = graph.global_vid(lvid);
    local_vertex_type vertex = graph.l_vertex(lvid);
    foreach(const procid_t& mirror, vertex.mirrors()) {
      vprog_exchange.send(mirror,
                          std::make_pair(vid, vertex_programs[lvid]),
                          thread_id);
    }
  } // end of sync_vertex_program



  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  recv_vertex_programs(const bool try_to_recv) {
    procid_t procid(-1);
    typename vprog_exchange_type::buffer_type buffer;
    while(vprog_exchange.recv(procid, buffer, try_to_recv)) {
      foreach(const vid_prog_pair_type& pair, buffer) {
        const lvid_type lvid = graph.local_vid(pair.first);
  //      ASSERT_FALSE(graph.l_is_master(lvid));
        vertex_programs[lvid] = pair.second;
        active_minorstep.set_bit(lvid);
      }
    }
  } // end of recv vertex programs


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  sync_vertex_data(lvid_type lvid, const size_t thread_id) {
    ASSERT_TRUE(graph.l_is_master(lvid));
    const vertex_id_type vid = graph.global_vid(lvid);
    local_vertex_type vertex = graph.l_vertex(lvid);
    foreach(const procid_t& mirror, vertex.mirrors()) {
      vdata_exchange.send(mirror, std::make_pair(vid, vertex.data()), thread_id);
    }
  } // end of sync_vertex_data





  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  recv_vertex_data(bool try_to_recv) {
    procid_t procid(-1);
    typename vdata_exchange_type::buffer_type buffer;
    while(vdata_exchange.recv(procid, buffer, try_to_recv)) {
      foreach(const vid_vdata_pair_type& pair, buffer) {
        const lvid_type lvid = graph.local_vid(pair.first);
        ASSERT_FALSE(graph.l_is_master(lvid));
        graph.l_vertex(lvid).data() = pair.second;
      }
    }
  } // end of recv vertex data


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  sync_gather(lvid_type lvid, const gather_type& accum, const size_t thread_id) {
    if(graph.l_is_master(lvid)) {
      vlocks[lvid].lock();
      if(has_gather_accum.get(lvid)) {
        gather_accum[lvid] += accum;
      } else {
        gather_accum[lvid] = accum;
        has_gather_accum.set_bit(lvid);
      }
      vlocks[lvid].unlock();
    } else {
      const procid_t master = graph.l_master(lvid);
      const vertex_id_type vid = graph.global_vid(lvid);
      gather_exchange.send(master, std::make_pair(vid, accum), thread_id);
    }
  } // end of sync_gather

  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  recv_gathers(const bool try_to_recv) {
    procid_t procid(-1);
    typename gather_exchange_type::buffer_type buffer;
    while(gather_exchange.recv(procid, buffer, try_to_recv)) {
      foreach(const vid_gather_pair_type& pair, buffer) {
        const lvid_type lvid = graph.local_vid(pair.first);
        const gather_type& accum = pair.second;
        ASSERT_TRUE(graph.l_is_master(lvid));
        vlocks[lvid].lock();
        if( has_gather_accum.get(lvid) ) {
          gather_accum[lvid] += accum;
        } else {
          gather_accum[lvid] = accum;
          has_gather_accum.set_bit(lvid);
        }
        vlocks[lvid].unlock();
      }
    }
  } // end of recv_gather


  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  sync_message(lvid_type lvid, const size_t thread_id) {
    ASSERT_FALSE(graph.l_is_master(lvid));
    const procid_t master = graph.l_master(lvid);
    const vertex_id_type vid = graph.global_vid(lvid);
    message_exchange.send(master, std::make_pair(vid, messages[lvid]), thread_id);
  } // end of send_message




  template<typename VertexProgram>
  void xadaptive_engine<VertexProgram>::
  recv_messages(const bool try_to_recv) {
	//xie insert
	size_t local_msgs = 0;
	
    procid_t procid(-1);
    typename message_exchange_type::buffer_type buffer;
    while(message_exchange.recv(procid, buffer, try_to_recv)) {
      foreach(const vid_message_pair_type& pair, buffer) {
        const lvid_type lvid = graph.local_vid(pair.first);
        ASSERT_TRUE(graph.l_is_master(lvid));
        vlocks[lvid].lock();

		//xie insert
		local_msgs++;
	
		if( has_message.get(lvid) ) {
          messages[lvid] += pair.second;
        } else {
          messages[lvid] = pair.second;
          has_message.set_bit(lvid);
        }
        vlocks[lvid].unlock();
      }
    }

	each_iteration_nmsg += local_msgs;
  } 



  
} ;// namespace

#include <graphlab/macros_undef.hpp>

#endif // GRAPHLAB_DISTRIBUTED_ENGINE_HPP

