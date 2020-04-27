import { Injectable } from '@angular/core';
import { cube, cube20, IScrambleRequest, ISolveResponse, ISolveRequest } from './rubiks/rubiks';
import { HttpService } from './http.service';


@Injectable({
  providedIn: 'root'
})
export class CommonService {

  status = {
    loading: true,
    hasData: false,
  };
  cuda: boolean;
  agents: string[];
  timeLimit: number;
  selectedAgent: number;
  hasSearchedForSolution = false;
  hasSolution: boolean;
  exploredStates: number;
  actionQueue: string[];
  statesLeft: cube[];
  searchedStates: number;

  state: cube;
  state20: cube20;
  actions = ["F", "f", "B", "b", "T", "t", "D", "d", "L", "l", "R", "r"];

  constructor(private httpService: HttpService) { }

  setSelectedHost(host: { name: string, address: string }) {
    this.httpService.selectedHost = host;
    this.getInfo();
    if (!this.status.hasData) {
      this.getSolved();
    }
  }

  public async getInfo() {
    try {
      this.status.loading = true;
      const { cuda, agents } = await this.httpService.getInfo();
      if (this.status.hasData) {
        this.status.loading = false;
        return;
      };
      this.cuda = cuda;
      this.agents = agents;
      this.selectedAgent = 0;
      this.timeLimit = 1;
      this.status.hasData = true;
      this.status.loading = false;
    } catch (e) {
      this.status.hasData = false;
      this.status.loading = false;
    }
  }

  public async getSolved(force = false) {
    if (this.status.loading && !force) return;
    this.status.loading = true;
    const { state, state20 } = await this.httpService.getSolved();
    this.state = state;
    this.state20 = state20;
    this.status.loading = false;
  }

  public async act(action: number) {
    if (this.status.loading) return;
    this.status.loading = true;
    const { state, state20 } = await this.httpService.performAction({action, state20: this.state20});
    this.state = state;
    this.state20 = state20;
    this.status.loading = false;
  }

  public async scramble(depth: number) {
    if (this.status.loading) return;
    this.status.loading = true;
    const scrambleRequest: IScrambleRequest = { depth, state20: this.state20, };
    const { states, finalState20 } = await this.httpService.scramble(scrambleRequest);
    this.statesLeft = states;
    this.state20 = finalState20;
    this.status.loading = false;
    this.animateStates();
  }

  public async solve() {
    if (this.status.loading) return;
    this.status.loading = true;
    const solveRequest: ISolveRequest = {
      agentIdx: this.selectedAgent,
      timeLimit: this.timeLimit,
      state20: this.state20,
    };
    const { solution, actions, searchedStates, states, finalState20 } = await this.httpService.solve(solveRequest);
    this.hasSearchedForSolution = true;
    this.hasSolution = solution;
    this.searchedStates = searchedStates;
    this.exploredStates = states.length;
    this.actionQueue = actions.map(val => this.actions[2*val[0] + val[1]]);
    this.state20 = finalState20;
    this.statesLeft = states;
    this.status.loading = false;
  }

  public step() {
    this.state = this.statesLeft.shift();
    if (this.actionQueue?.length) this.actionQueue.shift();
  }

  public async animateStates() {
    const frameLength = Math.min(1000 / this.statesLeft.length, 100);
    while (this.statesLeft.length) {
      const promise = new Promise((resolve, reject) => {
        setTimeout(() => resolve(), frameLength);
      })
      await promise;
      this.step();
    }
  }
}
