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
  solveActions: string[];

  state: cube;
  state20: cube20;
  actions = ["F", "f", "B", "b", "T", "t", "D", "d", "L", "l", "R", "r"];

  constructor(private httpService: HttpService) { }

  setSelectedHost(host: { name: string, address: string }) {
    this.httpService.selectedHost = host;
    this.getInfo();
  }

  public async getInfo() {
    try {
      this.status.loading = true;
      const { cuda, agents } = await this.httpService.getInfo();
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
    this.state20 = finalState20;
    await this.animateStates(states);
    this.status.loading = false;
  }

  public async solve() {
    if (this.status.loading) return;
    this.status.loading = true;
    const solveRequest: ISolveRequest = {
      agentIdx: this.selectedAgent,
      timeLimit: this.timeLimit,
      state20: this.state20,
    };
    const { solution, actions, states, finalState20 } = await this.httpService.solve(solveRequest);
    this.hasSearchedForSolution = true;
    this.hasSolution = solution;
    this.solveActions = actions.map(val => this.actions[2*val[0] + val[1]]);
    this.state20 = finalState20;
    if (this.hasSolution) {
      await this.animateStates(states);
    }
    this.status.loading = false;
  }

  private async animateStates(states: cube[]) {
    const frameLength = Math.min(1000 / states.length, 100);
    for (const state of states) {
      const promise = new Promise<cube>((resolve, reject) => {
        setTimeout(() => resolve(state), frameLength);
      })
      this.state = await promise;
    }
    console.log("Færdig med scarmbulering");
    return null;
  }
}
